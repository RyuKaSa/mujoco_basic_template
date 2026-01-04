from stable_baselines3 import PPO
from gymnasium_wrapper import Environment

# we create the environment from the gymnasium wrapper 
env = Environment()

# we create the policy, here, PPO, proximal policy optimization
model = PPO("MlpPolicy", env, verbose=1)


model.learn(total_timesteps=1000000)

model.save("PPO_Pendulum")
