# project

# Assignment-2

# Trajectory Tracking with Hindsight Experience Replay

Method is tested on [highway](https://highway-env.readthedocs.io/) goal conditioned continuous control task

### Overview

Train the model:
```
python code/run.py --test False
```
Test model:
```
python code/run.py --test True
```
You could define model type and which character to draw:
```
python code/run.py --test True --model_type HER --character Z
```
Settings can be adjusted with different arguments to run.py.

### Results

The left one is using DDPG, the right one is using HER+DDPG.

1. Z
<p float="left">
  <img src="video/ddpg-Z.gif" width="400" />
  <img src="video/her-Z.gif" width="400" /> 
</p>

2. L
<p float="left">
  <img src="video/ddpg-L.gif" width="400" />
  <img src="video/her-L.gif" width="400" /> 
</p>
3. U
<p float="left">
  <img src="video/ddpg-U.gif" width="400" />
  <img src="video/her-U.gif" width="400" /> 
</p>

4. all together

Due to time limitation, you can directly watch videos instead of gif.

The video for DDPG can be found [here](https://github.com/McGill-COMP-766-ECSE-683-Assignments/assignment-2-SHITIANYU-hue/blob/main/video/ddpgLUZ.mov) ;

The video for HER can be found [here](https://www.youtube.com/watch?v=7vsu0vh7vnA) ;

### Report

[Here](https://github.com/McGill-COMP-766-ECSE-683-Assignments/assignment-2-SHITIANYU-hue/blob/main/ECSE683assignment2_TianyuShi.pdf)

### Reference

Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., ... & Zaremba, W. (2017). Hindsight experience replay. In Advances in neural information processing systems (pp. 5048-5058)


