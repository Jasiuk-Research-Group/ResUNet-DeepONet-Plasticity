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

If you find our model helpful in your specific applications and researches, please cite our paper:
@article{he2023novel,
  title={Novel DeepONet architecture to predict stresses in elastoplastic structures with variable complex geometries and loads},
  author={He, Junyan and Koric, Seid and Kushwaha, Shashank and Park, Jaewan and Abueidda, Diab and Jasiuk, Iwona},
  journal={arXiv preprint arXiv:2306.03645},
  year={2023}
}


The training data is large in size and can be downloaded through the following UIUC Box link:
https://uofi.box.com/s/2ul0gtkyw8ziic5ka6kmvi0bj5yr4sjq
All three models described in the paper are provided.