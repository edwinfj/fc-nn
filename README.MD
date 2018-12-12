# FC-NN

A fully connected neural network with arbitrary layer numbers. Derived from 
[Stanford CS231n] assignments. The net is built from scratch. 
The architecture of the net is:

`input---[-affine-[-batch normalization-]-relu-[-dropout-]] x N---affine--softmax`

By default batch normalization and dropout are _**not**_ used. L2 weight decay
is used for regularization.

### [net_train_example.txt] demonstrates how to train a fully connected net.
The basic steps to do training:
1. create a net model, specifying the number of layers and number of units in each layer.
2. create a solver, which specifies the optimization method for training.

Options of optimization method:
- sgd
- sgd_momentum: Nesterov momentum
- rms_prop
- adam

[gradient_check_examples.txt] demonstrate how to do numerical check for backprop
gradients of a net.

[predict_test_example.txt] demonstrates how to run prediction using a trained 
net model, or test a trained model on a dataset.

### [Requirements]: python2.7, numpy, matplotlib, pillow


[Stanford CS231n]: http://cs231n.github.io/
[net_train_example.txt]: https://github.com/edwinfj/fc-nn/blob/master/net_train_example.txt
[gradient_check_examples.txt]: https://github.com/edwinfj/fc-nn/blob/master/gradient_check_examples.txt
[predict_test_example.txt]: https://github.com/edwinfj/fc-nn/blob/master/predict_test_example.txt
[Requirements]: https://github.com/edwinfj/fc-nn/blob/master/requirements.txt
