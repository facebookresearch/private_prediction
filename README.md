Private Prediction
========

This repository contains code that can be used to reproduce the experimental
results presented in the paper:

- L.J.P. van der Maaten* and A.Y. Hannun*. **[The Trade-Offs of Private Prediction](https://arxiv.org/abs/2007.05089)**. arXiv 2007.05089, 2020.

# Installation

The code requires Python 3.5+, [PyTorch 1.5.0+](https://github.com/pytorch/pytorch),
[torchvision 0.6.0+](https://github.com/pytorch/vision), and
[visdom](https://github.com/facebookresearch/visdom) (optional).
It also uses parts of [TensorFlow Privacy](https://github.com/tensorflow/privacy)
and [pytorch_resnet_cifar10](https://github.com/akamaster/pytorch_resnet_cifar10).

Presuming you have installed [Anaconda](https://docs.anaconda.com/anaconda/install/),
you can install all the dependencies via:
```
conda install -c pytorch pytorch torchvision
pip install visdom
python install.py
```

# Usage

All experiments can be run via the `private_prediction_experiment.py` script.
For example, to train and test a linear model on the MNIST dataset using loss
perturbation with privacy loss 1.0, you can use the following command:

```
python private_prediction_experiment.py \
    --dataset mnist \
    --method loss_perturbation \
    --epsilon 1.0
```

The following input arguments can be used to change the model, private prediction
method, and privacy loss:

- `--model`: the model used can be `linear` (default) or `resnet{20,32,44,56,110,1202}`
- `--method`: private prediction method can be `subsagg` (default), `loss_perturbation`, `{model,logit}_sensitivity`, or `dpsgd`
- `--epsilon`: privacy loss value for predictions (default = infinity)
- `--delta`: privacy failure probability for predictions (default = 0.0)
- `--inference_budget`: number of inferences to support (default = -1 to try many values)
- `--weight_decay`: L2-regularization parameter (default = 0.0; set to -1 to cross-validate)

The following input arguments can be used to change details of the optimizer:

- `--optimizer`: optimizer used can be `lbfgs` (default) or `sgd`
- `--num_epochs`: number of training epochs (default = 100)
- `--batch_size`: batch size for SGD optimization (default = 32)
- `--learning_rate`: initial learning rate for SGD optimization (default = 1.0)

The following input arguments alter hyperparameters of specific private prediction methods:

- `--num_models`: number of models in subsample-and-aggregate method (default = 32)
- `--noise_dist`: noise distribution used in sensitivity methods can be `sqrt_gaussian` (default), `laplacian`, `gaussian`, `advanced_gaussian`
- `--clip`: gradient clipping value for DP-SGD (default = 1e-1; set to -1 to cross-validate)
- `--use_lr_scheduler`: use learning rate reduction (for DP-SGD)

The following input arguments alter the dataset used for experimentation:

- `--dataset`: the dataset used can be `mnist` (default), `mnist1m`, `cifar10`, or `cifar100`
- `--num_samples`: number of training samples to use (default: all)
- `--num_classes`: number of classes to use (default: all)
- `--pca_dims`: number of PCA dimensions for data (default: PCA not used)

The following input arguments alter other system properties:

- `--device`: compute device can be `cpu` (default) or `gpu`
- `--visdom`: visdom server for learning curves (default = localhost)
- `--num_repetitions`: number of times to repeat experiment (default = 10)
- `--data_folder`: folder in which to store dataset for re-use
- `--result_file`: file in which to write experimental results (default: unused)

# Using the MNIST-1M Dataset

The MNIST-1M dataset used in the paper is not directly available for download,
but can be generated using [InfiniMNIST](https://leon.bottou.org/projects/infimnist).

Download [InfiniMNIST](https://leon.bottou.org/projects/infimnist) and run:
```
make
mkdir /tmp/infinimnist
infimnist patterns 70000 1069999 > /tmp/infinimnist/mnist1m-images-idx3-ubyte
infimnist labels 70000 1069999 > /tmp/infinimnist/mnist1m-labels-idx1-ubyte
infimnist patterns 0 9999 > t10k-images-idx3-ubyte
infimnist labels 0 9999 > t10k-labels-idx1-ubyte
```

Now, you should be able to run experiments on the MNIST-1M dataset, for example:
```
python private_prediction_experiment.py \
    --dataset mnist1m \
    --num_samples 100000 \
    --method loss_perturbation \
    --epsilon 1.0 \
    --data_folder /tmp/infinimnist
```

# Citing this Repository

If you use the code in this repository, please cite the corresponding paper:

- L.J.P. van der Maaten* and A.Y. Hannun*. **[The Trade-Offs of Private Prediction](https://arxiv.org/abs/2007.05089)**. arXiv 2007.05089, 2020.

# License

This code is released under a CC-BY-NC 4.0 license. Please see the [LICENSE](LICENSE) file for more information.

Please review Facebook Open Source [Terms of Use](https://opensource.facebook.com/legal/terms) and [Privacy Policy](https://opensource.facebook.com/legal/privacy).
