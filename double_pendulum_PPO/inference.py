import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from gymnasium_wrapper import Environment

model = PPO.load("PPO_Pendulum")

# create env
env = Environment()

# reset and get first observation
obs, _ = env.reset()

# run with viewer
with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    while viewer.is_running():
        # get action from trained policy
        action, _ = model.predict(obs, deterministic=True)
        
        # step environment
        obs, reward, terminated, truncated, _ = env.step(action)
        
        # sync viewer
        viewer.sync()
        
        # real-time
        time.sleep(env.model.opt.timestep)
        
        # reset if episode ends
        if terminated or truncated:
            obs, _ = env.reset()