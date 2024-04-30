import os

import torch
from torchvision import datasets, transforms
import torchvision.transforms.functional as tvf

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data import create_transform

from data.build import build_imagenet_dataset
import random
import numpy as np

def build_dataset(is_train, args):
    transform, post = build_transform(is_train, args)

    print("Transform = ")
    if isinstance(transform, tuple):
        for trans in transform:
            print(" - - - - - - - - - - ")
            for t in trans.transforms:
                print(t)
    else:
        for t in transform.transforms:
            print(t)
    print("---------------------------")
    if args.data == 'imagenet':
        dataset = build_imagenet_dataset(args, is_train, transform)
        nb_classes = 1000
    elif args.data == 'cifar10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    else:
        raise NotImplementedError()

    assert nb_classes == args.nb_classes
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes, post


def _gaussian_threat(sigma):
    def _threat(x):
        if sigma <= 0:
            return x
        noise = torch.randn_like(x)
        # adaptation to inception std (0.5)
        sigma_a = 2. * sigma
        # alpha = 1. / (sigma_a ** 2 + 1)
        # y = np.sqrt(alpha) * (x + sigma_a * noise)
        y = x + sigma_a * noise
        return y
    return _threat


def build_transform(is_train, args):
    if args.data == 'imagenet':
        im_size = 224
    elif args.data == 'cifar10':
        im_size = 32
    else:
        raise NotImplementedError()
    resize_im = im_size > 32
    # do_resize = im_size != args.input_size

    # imagenet_default_mean_and_std = args.imagenet_default_mean_and_std
    # mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
    # std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
    # if args.clip_mean_and_std:
    #     mean = [0.48145466, 0.4578275, 0.40821073]
    #     std = [0.26862954, 0.26130258, 0.27577711]

    # Always use Inception mean/std
    mean = IMAGENET_INCEPTION_MEAN
    std = IMAGENET_INCEPTION_STD
    if args.clip_mean_and_std:
        print("****** Use CLIP mean/std ******")
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    post = [_gaussian_threat(args.gaussian_std)]
    post.append(transforms.Resize(args.input_size, interpolation=3))  # to maintain same ratio w.r.t. 224 images
    post.append(transforms.CenterCrop(args.input_size))
    post = transforms.Compose(post)

    if is_train:
        # this should always dispatch to transforms_imagenet_train
        try:
            scale = (args.crop_scale, 1)
        except:
            scale = None
        print (f"use crop scale {scale}")
        transform = create_transform(
            input_size=im_size,
            is_training=True,
            scale=scale,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with RandomCrop
            transform.transforms[0] = transforms.RandomCrop(im_size, padding=4)
        return transform, post

    t = []
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    t.append(post)

    return transforms.Compose(t), post


