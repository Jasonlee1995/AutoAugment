# AutoAugment Implementation with Pytorch
- Unofficial implementation of the paper *AutoAugment: Learning Augmentation Strategies From Data*


## 0. Develop Environment
```
Docker Image
- pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
```
- Using Single GPU


## 1. Implementation Details
- model.py : Wide ResNet model
- train.py : train Wide ResNet
- utils.py : count correct prediction
- AutoAugment - Wide ResNet 28 10 - Cifar 10 (Single) : install library, download dataset, preprocessing, train and result
- AutoAugment - Wide ResNet 28 10 - Cifar 10 (5 Average) : average 5 runs
- Details
  * Follow Cutout train details
    * batch size 128, learning rate 0.1, nesterov momentum 0.9, weight decay 0.0005
    * learning rate schedulers on [60, 120, 160] with 0.2 learning rate drop
    * augmentation : mean/std preprocessing, pad and crop
  * Policy from paper and policy from code are different
  * Use ImageNet policy to check transfer


## 2. Result Comparison on CIFAR-10
|Source|Score|Detail|
|:-:|:-:|:-|
|Paper|97.4|Policy from CIFAR-10 with model WRN 28-10|
|Current Repo|97.27|Policy from CIFAR-10 policy from official code with model WRN 28-10|
|Current Repo|97.27|Policy from CIFAR-10 policy from paper with model WRN 28-10|
|Current Repo|96.96|Policy from ImageNet policy from paper with model WRN 28-10|


## 3. Reference
- AutoAugment: Learning Augmentation Strategies From Data [[paper]](https://arxiv.org/pdf/1805.09501.pdf) [[official code]](https://github.com/tensorflow/models/tree/master/research/autoaugment)
