# SAiDL-assignment
This repository is the Summer Induction Assignment of Society for Artificial Intelligence and Deep Learning for the year 2021.

The attempted sections are:

1. [Reinforcement Learning](https://github.com/soham-chitnis10/SAiDL-assignment/tree/main/RL)
2. [Computer Vision](https://github.com/soham-chitnis10/SAiDL-assignment/tree/main/Computer%20Vision)

## Reinforcement Learning

The aim is to find minima of the functions of type ![alt_text](http://www.sciweavers.org/tex2img.php?eq=ax%5E2%20%2B%20by%5E2%20%2B%20cxy%20%2B%20dx%20%2B%20ey%20%2B%20f&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
using policy gradient with Gaussian Distribution.

Total three reward functions were implemented:

1. Inverse of the distance from the minima of the current state which only depends on the state.
2. Exponent of the sum of the product of the difference of state from the minima and action along that directions. The Reward function is ![alt text](http://www.sciweavers.org/tex2img.php?eq=%20e%5E%7B%5Csum_%7Bi%20%3D0%7D%5E%7B1%7D%20%20%5Cbig%28s_%7Bt_%7Bi%7D%7D%20-%20x_%7B0_%7Bi%7D%7D%5Cbig%29%20.%20a_%7Bt_%7Bi%7D%7D%7D%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)
3. The reward function is given by ![alt_text](http://www.sciweavers.org/tex2img.php?eq=%20%20R%28s_%7Bt%7D%2Ca_%7Bt%7D%29%20%20%3D%20%5Csum_%7Bi%3D0%7D%5E%7B1%7D%20%5Cbegin%7Bcases%7D%5Cbig%28s_%7Bt_%7Bi%7D%7D%20-%20x_%7B0_%7Bi%7D%7D%20%5Cbig%29.a_%7Bt_%7Bi%7D%7D%20%26%20%20%5Cbig%28s_%7Bt_%7Bi%7D%7D%20-%20x_%7B0_%7Bi%7D%7D%20%5Cbig%29.a_%7Bt_%7Bi%7D%7D%20%20%3E%200%5C%5C0%20%5Cend%7Bcases%7D%20%20%20&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0)

To run the environment and agent run the ```main.py``` which can be found [here](https://github.com/soham-chitnis10/SAiDL-assignment/blob/main/RL/main.py). The environment is a custom OpenAI environment which can found [here](https://github.com/soham-chitnis10/SAiDL-assignment/blob/main/RL/Quadratic_2D_env.py).

All three functions are also implemented in colab so colab notebooks can be found in the same [folder](https://github.com/soham-chitnis10/SAiDL-assignment/tree/main/RL)

The agent was trained for 3000 episodes for reward functions 1 and 3. The agent was tested for 1000 episodes. Reward function 2 could not be trained more than 1000 due to overflow in exponent ,therefore, testing was not conducted on reward function 2. A successful episode was defined as the absolute differnce of particular coordinate with that state is less than 0.1. The graphs of Inverse of the distance of the last state from minima was plotted against the episodes. The peak of the graph would indicate successful episodes.

Reward function 1: Successful Training Episodes: 18/3000 Successful Testing Episodes: 6/1000

Reward function 3: Successful Training Episodes: 28/3000 Successful Testing Episodes: 6/1000

## Computer Vision
