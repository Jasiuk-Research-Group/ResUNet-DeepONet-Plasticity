# ResUNet-DeepONet-Plasticity
Implementation of a ResUNet-based DeepONet for predicting stress distribution on variable input geometries subject to variable loads. A ResUNet is used in the trunk  network to encode the variable input geometries, and a feed-forward neural network is used in the branch to encode the loading parameters.


The DeepONet implementation and training is based on DeepXDE:
@article{lu2021deepxde,
  title={DeepXDE: A deep learning library for solving differential equations},
  author={Lu, Lu and Meng, Xuhui and Mao, Zhiping and Karniadakis, George Em},
  journal={SIAM review},
  volume={63},
  number={1},
  pages={208--228},
  year={2021},
  publisher={SIAM}
}

The implementation of the ResUNet is adapted from Jan Palase:
https://github.com/JanPalasek/resunet-tensorflow

If you find our model helpful in your specific applications and researches, please cite this article as: 
J. He, S. Koric, S. Kushwaha et al., Novel DeepONet architecture to predict stresses in elastoplastic structures with variable complex geometries and loads,Computer Methods in Applied Mechanics and Engineering (2023) 116277, https://doi.org/10.1016/j.cma.2023.116277.


The training data is large in size and can be downloaded through the following UIUC Box link:
https://uofi.box.com/s/2ul0gtkyw8ziic5ka6kmvi0bj5yr4sjq
All three models described in the paper are provided.