"""
#!/bin/bash

#SBATCH --nodes 1
#SBATCH --gres gpu:1 
#SBATCH --time=48:00:00
#SBATCH --mem=50GB
"""

source activate Rl
KERAS_BACKEND=tensorflow

agent="DQV"
exploration="e-greedy"
game="PongDeterministic-v4"
v_activation="relu"

nohup python DQV_learning.py --agent $agent --game $game --exploration $exploration --final_v_activation $v_activation &> Log_Pong_experiment.out
