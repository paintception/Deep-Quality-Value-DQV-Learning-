# Deep Quality Value (DQV) Learning
DQV-Learning: a novel faster and stable synchronous Deep Reinforcement Learning algorithm.

![](https://user-images.githubusercontent.com/14283557/46071343-165d7b80-c180-11e8-8b23-37cfecb96534.jpg)

This repo contains the code and models of the paper Deep Quality-Value (DQV) Learning:(https://arxiv.org/abs/1810.00368)

Given a standard RL setting, unlike Temporal-Difference (TD) Reinforcement Learning algorithms such as DQN and DDQN, DQV aims to learn directly a Value function with the TD(λ) update rule:

![](https://latex.codecogs.com/gif.latex?V%28s_t%29%3A%3D%20V%28s_t%29%20&plus;%20%5Calpha%20%5Cbig%5B%20r_%7Bt%7D%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%29%20-%20V%28s_t%29%20%5Cbig%5D)

which estimates can then be used to learn the state-action pairs of the Q-function via:
![](https://latex.codecogs.com/gif.latex?Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%3A%3D%20Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%20&plus;%20%5Calpha%20%5Cbig%5Br_%7Bt%7D%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%29%20-%20Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%29%20%5Cbig%5D)

When approximating the two update rules with a non parametric function approximator like a neural network, both functions can be expressed as regression problems that can be minimized through gradient descent by optimizing

![](https://latex.codecogs.com/gif.latex?L_%7B%5CPhi%7D%20%3D%20%5Cmathds%7BE%7D%20%5Cbig%5B%28r_%7Bt%7D%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%2C%20%5CPhi%29%20-%20V%28s_%7Bt%7D%2C%20%5CPhi%29%29%5E%7B2%7D%5Cbig%5D%2C)

and

![](https://latex.codecogs.com/gif.latex?L_%7B%5Ctheta%7D%20%3D%20%5Cmathds%7BE%7D%20%5Cbig%5B%28r_%7Bt%7D%20&plus;%20%5Cgamma%20V%28s_%7Bt&plus;1%7D%2C%20%5CPhi%29%20-%20Q%28s_%7Bt%7D%2C%20a_%7Bt%7D%2C%20%5Ctheta%29%29%5E%7B2%7D%5Cbig%5D)

To check the benefits of these update rules that are able to learn up to 3 times faster and better than DQN and DDQN as presented in the paper be sure to:

  * Install the requirements list in your favourite virtualenv with `pip install -r requirements.txt`

  * The directory `./models/` contains the weights of the Value and Quality networks which have obtained the results that are reported in the paper. If you aim to test one of these pre-trained networks you can run the `./models/DQV_Trained.py` script. Be sure to use the weights that match the appropriate Open-Ai environment. 
  
  * If you aim to train a model from scratch, or adapt DQV on a different DRL problem, you can find the code for training the agents in `./train/`. The `.train_job.sh` script allows you to choose which Open-Ai environment you want to train your agents on, and which exploration strategy to follow. So far the code only supports e-greedy and Maxwell-Boltzman exploration. The script will call `DQV_learning.py` which is the code that has been used to train the networks that are present in `./models/`, and that matches with what is reported in the paper. Training i.e. on the game Pong lasts ≈ 8 hours on a GTX 1080 GPU machine and the environment should be solved in ≈ 400 episodes.
