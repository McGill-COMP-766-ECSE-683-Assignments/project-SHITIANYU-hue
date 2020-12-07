import gym
import highway_env
from rl_agents.agents.common.factory import agent_factory
from rl_agents.trainer.evaluation import Evaluation
import numpy as np
# from utils import record_videos, show_videos, capture_intermediate_frames
import math
# Make environment
env = gym.make("highway-v0")
obs, done = env.reset(), False

# Make agent
agent_config = {
    "__class__": "<class 'rl_agents.agents.tree_search.deterministic.DeterministicPlannerAgent'>",
    "env_preprocessors": [{"method":"simplify"}],
    "budget": 50,
    "gamma": 0.7,
}
agent = agent_factory(env, agent_config)
max_timesteps=10
# Run episode
maxs=[]
mins=[]
for episode in range(50):
    obs, done = env.reset(), False
    episode_reward=0
    for t in range(max_timesteps):
        action = agent.act(obs)
        agent.write_tree()
        next_obs, reward, done, info = env.step(action)
        position1=np.min(next_obs[1:5],axis=0)
        distance1=math.sqrt(position1[1]**2+position1[2]**2)
        maxs.append(distance1)
        position2=np.max(next_obs[1:5],axis=0)
        distance2=math.sqrt(position2[1]**2+position2[2]**2)
        mins.append(distance2)
        # print('min',distance1)
        # print('max',distance2)
        # agent.record(obs, action,reward, next_obs,done, info)
        obs=next_obs
        episode_reward+=reward
        # print('reward',reward)
        env.render()
    # agent.update()
    # print('episode:',episode,'episode reward:',episode_reward)
# np.save('max.npy',maxs)
# np.save('min.npy',mins)
env.close()
# evaluation = Evaluation(env,
#                         agent,
#                         run_directory=None,
#                         num_episodes=1,
#                         sim_seed=2,
#                         recover=None,
#                         display_env=False,
#                         display_agent=False,
#                         display_rewards=True)
# evaluation.train()
