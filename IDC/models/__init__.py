from .convnet import ConvNet
from .resnet import ResNet
from .resnet_ap import ResNetAP
from .densenet import DenseNet121, DenseNet161, DenseNet169, DenseNet201, densenet_cifar
from .loss_optim_scheduler import *

def get_model(s):
    return {"convnet": ConvNet,
            "resnet": ResNet,
            "resnetap": ResNetAP,
            "densenet121": DenseNet121,
            "densenet161": DenseNet161,
            "densenet169": DenseNet169,
            "densenet201": DenseNet201,
            "densenet_cifar": densenet_cifar,
            }[s.lower()]

def get_loss(s):
    return {
        'l1': l1,
        'l2': l2,
        'bce': bce,
        'ce': ce
    }[s.lower()]


def get_optim(s):
    return {
        'adam': adam,
        'sgd': sgd,
        'adagrad': adagrad,
        'rmsprop': rmsprop,
    }[s.lower()]


def get_scheduler(s):
    return {
        'steplr': steplr,
        'multisteplr': multisteplr,
        'cosineannealinglr': cosineannealinglr,
        'reducelronplateau': reducelronplateau,
        'lambdalr': lambdalr,
        'cycliclr': cycliclr,
    }[s.lower()]
