"""
Manual Drone Control for Recording Demonstrations

Controls (while viewer window is focused):
    Z/S     - Forward/Backward
    Q/D     - Left/Right  
    E       - Up
    A       - Down
    
    X       - Reset episode (discard)
    C       - Pause/Resume recording
    ESC     - Save and quit

Records (observation, action) pairs for Behavioral Cloning.
A yellow arrow shows the direction from the drone to the current target.

Usage:
    pip install pynput
    python record_demo.py --episodes 10 --output demos.npz
    
Then train with:
    python train.py --bc --demo demos.npz
"""

import argparse
import numpy as np
import time
import threading
from collections import defaultdict

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    print("ERROR: mujoco not installed. Run: pip install mujoco")
    exit(1)

try:
    from pynput import keyboard
except ImportError:
    print("ERROR: pynput not installed. Run: pip install pynput")
    exit(1)

from drone_env import DroneEnv, quat_to_euler


def rotation_matrix_from_z_to_direction(direction):
    """Create rotation matrix that rotates z-axis to given direction."""
    d = np.asarray(direction, dtype=np.float64)
    norm = np.linalg.norm(d)
    if norm < 1e-8:
        return np.eye(3)
    d = d / norm
    z_axis = np.array([0.0, 0.0, 1.0])
    
    # Handle case where direction is parallel to z-axis
    dot = np.dot(d, z_axis)
    if dot > 0.9999:
        return np.eye(3)
    elif dot < -0.9999:
        return np.diag([1.0, -1.0, -1.0])
    
    # Rodrigues' rotation formula
    v = np.cross(z_axis, d)
    s = np.linalg.norm(v)
    c = dot
    
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])
    
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s * s + 1e-8))
    return R


def add_direction_arrow(viewer, start_pos, end_pos, 
                        rgba=[1.0, 0.8, 0.0, 0.8],
                        radius=0.06):
    """
    Add a visual arrow from start_pos pointing to end_pos.
    Uses viewer.user_scn for visualization-only geometry (no physics).
    """
    start = np.asarray(start_pos, dtype=np.float64)
    end = np.asarray(end_pos, dtype=np.float64)
    direction = end - start
    length = np.linalg.norm(direction)
    
    if length < 0.5:
        return  # Too short to draw
    
    direction_norm = direction / length
    
    # Arrow starts exactly at drone, full length to target
    arrow_center = start + direction_norm * (length / 2)
    
    # Rotation matrix to align z-axis with direction
    R = rotation_matrix_from_z_to_direction(direction_norm)
    mat = R.flatten()  # Row-major 3x3 matrix
    
    # Add arrow using built-in ARROW geom
    # For mjGEOM_ARROW: size = [shaft_radius, head_radius, half_length]
    if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
        geom = viewer.user_scn.geoms[viewer.user_scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_ARROW,
            [radius, radius * 2.5, length / 2],
            arrow_center,
            mat,
            rgba
        )
        viewer.user_scn.ngeom += 1


class KeyboardController:
    """Tracks keyboard state for drone control (AZERTY layout)."""
    
    def __init__(self):
        self.keys_pressed = set()
        self.lock = threading.Lock()
        self.should_quit = False
        self.should_reset = False
        self.paused = False
        
        # Start listener
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self.listener.start()
    
    def _on_press(self, key):
        with self.lock:
            try:
                self.keys_pressed.add(key.char.lower())
            except AttributeError:
                # Special keys
                if key == keyboard.Key.esc:
                    self.should_quit = True
    
    def _on_release(self, key):
        with self.lock:
            try:
                self.keys_pressed.discard(key.char.lower())
            except AttributeError:
                pass
    
    def check_special_keys(self):
        """Check for reset/pause commands (call once per frame)."""
        with self.lock:
            if 'x' in self.keys_pressed:
                self.keys_pressed.discard('x')
                self.should_reset = True
            if 'c' in self.keys_pressed:
                self.keys_pressed.discard('c')
                self.paused = not self.paused
                print(f"Recording {'PAUSED' if self.paused else 'RESUMED'}")
    
    def get_action(self, speed=1.0):
        """Convert current key state to action [vx, vy, vz]. AZERTY layout."""
        with self.lock:
            keys = self.keys_pressed.copy()
        
        vx, vy, vz = 0.0, 0.0, 0.0
        
        # Forward/backward (Z/S on AZERTY)
        if 'z' in keys:
            vx = speed
        if 's' in keys:
            vx = -speed
        
        # Left/right (Q/D on AZERTY)
        if 'q' in keys:
            vy = speed
        if 'd' in keys:
            vy = -speed
        
        # Up/down (E/A - avoiding space/shift which conflict with viewer)
        if 'e' in keys:
            vz = speed
        if 'a' in keys:
            vz = -speed
        
        return np.array([vx, vy, vz], dtype=np.float32)
    
    def stop(self):
        self.listener.stop()


class DemoRecorder:
    """Records demonstrations as (obs, action, next_obs, done) tuples."""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.next_observations = []
        self.dones = []
        
        # Current episode buffer
        self._ep_obs = []
        self._ep_acts = []
        self._ep_next_obs = []
        self._ep_dones = []
        
        self.episodes_completed = 0
        self.total_transitions = 0
    
    def add_transition(self, obs, action, next_obs, done):
        """Add a single transition to current episode."""
        self._ep_obs.append(obs.copy())
        self._ep_acts.append(action.copy())
        self._ep_next_obs.append(next_obs.copy())
        self._ep_dones.append(done)
    
    def end_episode(self, success=False):
        """Finalize current episode."""
        if len(self._ep_obs) > 0:
            self.observations.extend(self._ep_obs)
            self.actions.extend(self._ep_acts)
            self.next_observations.extend(self._ep_next_obs)
            self.dones.extend(self._ep_dones)
            
            self.total_transitions += len(self._ep_obs)
            self.episodes_completed += 1
            
            status = "SUCCESS" if success else "ended"
            print(f"Episode {self.episodes_completed} {status}: {len(self._ep_obs)} transitions "
                  f"(total: {self.total_transitions})")
        
        # Clear episode buffer
        self._ep_obs = []
        self._ep_acts = []
        self._ep_next_obs = []
        self._ep_dones = []
    
    def discard_episode(self):
        """Discard current episode without saving."""
        n = len(self._ep_obs)
        self._ep_obs = []
        self._ep_acts = []
        self._ep_next_obs = []
        self._ep_dones = []
        print(f"Episode discarded ({n} transitions)")
    
    def save(self, filepath):
        """Save all demonstrations to npz file."""
        if self.total_transitions == 0:
            print("No transitions to save!")
            return
        
        np.savez(
            filepath,
            observations=np.array(self.observations, dtype=np.float32),
            actions=np.array(self.actions, dtype=np.float32),
            next_observations=np.array(self.next_observations, dtype=np.float32),
            dones=np.array(self.dones, dtype=bool),
        )
        print(f"Saved {self.total_transitions} transitions from {self.episodes_completed} episodes to {filepath}")


def update_target_marker(env):
    """Move the target marker mocap body to current target position."""
    try:
        # Get mocap body index for target_marker
        mocap_id = env.model.body("target_marker").mocapid[0]
        env.data.mocap_pos[mocap_id] = env.target_pos.copy()
    except Exception as e:
        print(f"Warning: Could not update target marker: {e}")


def record_demos(args):
    print("=" * 60)
    print("DEMONSTRATION RECORDING")
    print("=" * 60)
    print("""
Controls (AZERTY):
    Z/S     - Forward/Backward
    Q/D     - Left/Right
    E/A     - Up/Down
    
    X       - Reset (discard current episode)
    C       - Pause/Resume recording
    ESC     - Save and quit

Target: Fly to the GREEN CUBE and touch it.
A YELLOW ARROW shows the direction to the target.
""")
    print("=" * 60)
    
    # Create environment
    env = DroneEnv()
    env.easy_ratio = args.easy_ratio
    env.min_target_dist = args.min_dist
    env.max_target_dist = args.max_dist
    env.dwell_time = args.dwell
    env.target_timeout = args.timeout
    
    # Reset to get model/data
    obs, _ = env.reset(seed=args.seed)
    update_target_marker(env)
    
    # Setup keyboard and recorder
    kb = KeyboardController()
    recorder = DemoRecorder()
    
    print(f"\nStarting... (target episodes: {args.episodes})")
    print(f"Target distance range: [{args.min_dist}, {args.max_dist}] m")
    print(f"Speed multiplier: {args.speed}")
    
    # Verify first target is at proper distance
    first_dist = np.linalg.norm(env.target_pos - env.data.qpos[:3])
    print(f"Drone position: {env.data.qpos[:3]}")
    print(f"Target position: {env.target_pos}")
    print(f"First target distance: {first_dist:.1f}m")
    
    if first_dist < 1.0:
        print("WARNING: Target too close! Check _spawn_target()")
    
    print("\nFocus the viewer window and use ZQSD + E/A to fly!\n")
    
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            first_sync = True
            episode = 0
            step = 0
            last_print = time.time()
            
            while viewer.is_running() and not kb.should_quit:
                if recorder.episodes_completed >= args.episodes:
                    print(f"\nReached {args.episodes} episodes!")
                    break
                
                if first_sync:
                    # Find the "track" camera by name using MuJoCo API
                    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
                    
                    if cam_id >= 0:
                        print(f"Found 'track' camera with id {cam_id}")
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                        viewer.cam.fixedcamid = cam_id
                    else:
                        print("Warning: 'track' camera not found, using default view")
                    
                    viewer.sync()
                    first_sync = False

                # Check for reset/pause
                kb.check_special_keys()
                
                if kb.should_reset:
                    kb.should_reset = False
                    recorder.discard_episode()
                    obs, _ = env.reset()
                    update_target_marker(env)
                    step = 0
                    continue
                
                # Get action from keyboard
                action = kb.get_action(speed=args.speed)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                # Record if not paused
                if not kb.paused:
                    recorder.add_transition(obs, action, next_obs, done)
                
                # Clear user scene and add direction arrow
                viewer.user_scn.ngeom = 0  # Reset custom geometry count
                drone_pos = env.data.qpos[:3].copy()
                add_direction_arrow(viewer, drone_pos, env.target_pos)
                
                # Update viewer
                viewer.sync()
                
                # Print status periodically
                if time.time() - last_print > 0.5:
                    pos = env.data.qpos[:3]
                    vel = env.data.qvel[:3]
                    dist = info['distance']
                    yaw = np.degrees(quat_to_euler(env.data.qpos[3:7])[2])
                    
                    status = "PAUSED" if kb.paused else "REC"
                    in_zone = ">>> IN TARGET ZONE <<<" if info['in_target_zone'] else ""
                    
                    print(f"[{status}] Ep {recorder.episodes_completed+1}/{args.episodes} | "
                          f"dist: {dist:.1f}m | vel: ({vel[0]:.1f}, {vel[1]:.1f}, {vel[2]:.1f}) | "
                          f"act: ({action[0]:.1f}, {action[1]:.1f}, {action[2]:.1f}) {in_zone}")
                    last_print = time.time()
                
                obs = next_obs
                step += 1
                
                # Update marker if target changed (was reached)
                if info.get('target_reached_this_step', False):
                    update_target_marker(env)
                
                # Episode ended
                if done:
                    success = info.get('target_reached_this_step', False) or info.get('targets_reached', 0) > 0
                    recorder.end_episode(success=success)
                    obs, _ = env.reset()
                    update_target_marker(env)
                    step = 0
                
                # Timing
                time.sleep(env.dt)
    
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        kb.stop()
        
        # Save any remaining episode
        if len(recorder._ep_obs) > 0:
            recorder.end_episode(success=False)
        
        # Save demonstrations
        if recorder.total_transitions > 0:
            recorder.save(args.output)
        
        env.close()
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Record demonstrations for behavioral cloning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to record")
    parser.add_argument("--output", type=str, default="demos.npz",
                        help="Output file for demonstrations")
    parser.add_argument("--speed", type=float, default=0.7,
                        help="Control speed multiplier (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    # Environment settings (should match training)
    parser.add_argument("--min-dist", type=float, default=5.0,
                        help="Minimum target distance")
    parser.add_argument("--max-dist", type=float, default=15.0,
                        help="Maximum target distance")
    parser.add_argument("--dwell", type=float, default=0.0,
                        help="Dwell time at target (0 = instant capture)")
    parser.add_argument("--timeout", type=float, default=60.0,
                        help="Episode timeout (longer for manual control)")
    parser.add_argument("--easy-ratio", type=float, default=1.0,
                        help="Ratio of easy (cardinal) targets")
    
    args = parser.parse_args()
    record_demos(args)