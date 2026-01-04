import mujoco
import gymnasium as gym
from gymnasium import spaces
import numpy as np


# inherit the env from gymnasium
class Environment(gym.Env):

    # run once 
    def __init__(self):

        # define model and data like in template simulation
        self.model = mujoco.MjModel.from_xml_path("model.xml")
        self.data = mujoco.MjData(self.model)

        # define observation space,
        # the input of the model
        # its gonna be angles and velocities, from data, which is the dynamic state,
        # but also need to give range
        self.observation_space = spaces.Box(
            # if unsure of bounds, use np.inf
            low=-np.inf, 
            high=np.inf, 
            # shape 6 for 2 angles (sin and cos) and their 2 velocities
            shape=(6,), 
            dtype=np.float32
        )

        # define action space
        # here, single torque value, the model only needs to control the singular motor
        self.action_space = spaces.Box(
            # match xml actuator set up which is ctrlrange="-1 1"
            low=-1.0, 
            high=1.0, 
            # just size 1, since a single motor input
            shape=(1,), 
            dtype=np.float32
        )


    # reset for new episode, we ann randomization for the ref angles, to generalize the model
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # reset sim state
        mujoco.mj_resetData(self.model, self.data)

        # add randomization for better learning experience on each new episode
        # we basically just want to set a random float between 1 pi and -1 pi
        # we use the gym random uniform, since the seed comes from gym, its reproducible, unlike np.random, even with same seed
        self.data.qpos[0] = self.np_random.uniform(-0.5, 0.5)
        self.data.qpos[1] = self.np_random.uniform(-0.5, 0.5)

        # return observation 
        return self._get_obs(), {}
    

    # physics step, which is mujoco setup, weaved with gymnasium exposure
    def step(self, action):
        # apply action from model to actuator
        self.data.ctrl[0] = action[0]

        # simulate physics ste^using model output
        mujoco.mj_step(self.model, self.data)

        # get new observation from fresh simulated step
        obs = self._get_obs()

        # compute rewards based on new observations
        # here, maximize reward for straight angles, cos(0); since the initial pose is oriented towards top already
        reward = 0
        for i in range(len(self.data.qpos)):
            angle = self.data.qpos[i]
            if -0.5 < angle < 0.5:
                reward += np.cos(angle)
            else:
                reward += np.cos(angle) - 1

        # check for loosing condition ? time limit, etc
        # no termination condition, it just swings, but loosing con would be coded here
        terminated = False
        # truncated means cut short, eg: took too long, time limit
        truncated = self.data.time > 30

        # expose observation of fresh step, and computed rewards, and if terminated 
        return obs, reward, terminated, truncated, {}
    

    def _get_obs(self):
        # return array format of observable,
        # here return both angles and velocities of hinges, first hinge happens to be the actuator motor
        return np.array([
            # angle values
            np.sin(self.data.qpos[0]),
            np.cos(self.data.qpos[0]),
            np.sin(self.data.qpos[1]),
            np.cos(self.data.qpos[1]),
            # velocity values
            self.data.qvel[0],
            self.data.qvel[1]
        ], dtype=np.float32)
