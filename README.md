# Solving the Open AI gym's Robotics environment (FetchPickAndPlace-v1) using DDPG and HER

![gif](https://media.giphy.com/media/kfR5iyQgmq7PoiFTAf/giphy.gif)



NOTES:

-lowLevelTf_ddpg_her.py is the functional file. Based completely on [Buntyke's implementation of DDPG and HER](https://github.com/buntyke/her/blob/master/ddpg_her.py).

-ddpg_her.py and network_classes.py were an attempt to implement the same above code using high-level (tensorflow) Keras API. Had to discontinue it as the loss function of critic depended on the prediction from the same critic-taeget network which is hard to code using Keras.

-The corresponding graph images and the saved checkpoints are from the lowLevelTf_ddpg_her.py.

- ignore the trial* files
