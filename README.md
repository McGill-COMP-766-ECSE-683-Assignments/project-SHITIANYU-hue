# Final project

# Safe and Efficient Offline Reinforcement Learning 

### Abstract
Many practical online reinforcement learning applications, such as autonomous driving, will involve interactions with environment, which will have high deployment cost. In this work, we propose an offline reinforcement learning approach to learn from the collected dataset. Due to the conservativeness of most off-line reinforcement learning method, we introduce an exploration strategy to make the agent to explore efficiently in the environment. Meanwhile, we also introduce the Lyapunov functions to provide the safety guarantee during the policy learning process. An efficient and safe off-line reinforcement learning method is designed to allow the agent can not only efficiently explore the action space but also guarantee safety. Numerous experiments have been conducted in autonomous driving tasks to evaluate the learned policy. The experimental results indicate that the proposed model outperforms the baseline method.

### Environment

Method is tested on [highway](https://highway-env.readthedocs.io/)  continuous control tasks in [OpenAI gym](https://github.com/openai/gym). 
Networks are trained using [PyTorch 1.4](https://github.com/pytorch/pytorch) and Python 3.7. 


### Overview

Collect the data(e.g. for highway scenario):
```
python collect_data/collect_highway.py 
```
The collected data (state, action, reward, next_state, dones) are saved in:
```
buffers/
```
In stable_dynamics.py we define how to calculate the LF risk and in BCQ_g.py we define the exploration strategy and how we jointly optimize them. You could run our model by:
```
python code/g_main.py 
```
Settings can be adjusted with different arguments to g_main.py.

### Main Baselines

**Noisy  BCQ**:  In  this  version,  we  consider  only  adding exploration  strategy  on  the  policy  from BCQ framework. However, it doesn’t have any safety guarantee. This version is more likely to explore even for unsafe actions.

**Ours**:  An  efficient  and  safe  off-line  reinforcement learning  method  to  allow  the  agent  can  not  only efficiently  explore  the  action  space  but  also  guarantee  safety.


### Results

The left one is using our method , the right one is using Noisy BCQ.

1. **Highway scenario** : we found that our method has relative bigger minimum distance than Noisy BCQ while Noisy BCQ's policy is  more aggressive.

<p float="left">
  <img src="video/h-safe.gif" width="400" />
  <img src="video/h-unsafe.gif" width="400" /> 
</p>

2. **Parking scenario** : we found that our method will have small steering wheel angle and tend to be stable around the target point while Noisy BCQ will have sharper steering wheel agnle and larger acceleration which is more likely to oscilitate around the target point.

<p float="left">
  <img src="video/p-safe.gif" width="400" />
  <img src="video/p-unsafe.gif" width="400" /> 
</p>

3. Visulization of the state action visitation density
<p float="left">
  <img src="video/state-action-visitation.png" width="900" />
</p>
We  can  see  from  the  state  action visitation density  plot  that the original BCQ method tends to explore very cautiously with  very  limited  amount  state  and  action  visitations.   On  the other  hand,  the  noisy  BCQ  demonstrates  more  diverse  state and  action  visitation,  even  it  will  explore  some  action  that  are not common given the same states as BCQ. As a result, the noisy BCQ explores the state action space more efficiently than BCQ. Furthermore, our method can explore nearly the same state range as noisy BCQ but it will tend to explore the action space in a reasonable range, which is due to the safety concern within  our  method.  Combined  with  the  previous  experiment results, we can conclude that our method can achieve the best balance between safety and efficiency among BCQ and noisy BCQ.

### Report

[Here](https://github.com/McGill-COMP-766-ECSE-683-Assignments/project-SHITIANYU-hue/blob/main/ecse683report.pdf)

### Contact

Tianyu Shi: tianyu.shi3@mail.mcgill.ca



