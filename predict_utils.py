import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness', 'contrast',
    'elastic_transform', 'pixelate', 'jpeg_compression'
]
CORRUPTIONS_PER_TYPE = {
    'noise': ['gaussian_noise', 'shot_noise', 'impulse_noise'],
    'blur': ['defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur'],
    'weather': ['snow', 'frost', 'fog', 'brightness'],
    'digital': ['contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
}


def load_cifar10c(root, corruption, transform):
    data_dir = Path(root)
    base_path = data_dir / 'CIFAR-10-C'
    ds = datasets.CIFAR10(root, train=False, transform=transform, download=True)

    # Reference to original data is mutated
    ds.data = np.load(base_path / f"{corruption}.npy")
    ds.targets = torch.LongTensor(np.load(base_path / f"labels.npy"))
    return ds


class TransformTensorDataset(TensorDataset):
    """TensorDataset with support of transforms."""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        if self.transform:
            x = self.transform(x)
        y = self.tensors[1][index]
        return x, y

    def __len__(self):
        return self.tensors[0].size(0)


class NormalizeLayer(torch.nn.Module):
    def __init__(self, mean, std):
        super(NormalizeLayer, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def forward(self, input: torch.tensor):
        _device = input.device
        mean = self.mean.to(_device).view(-1, 1, 1)
        std = self.std.to(_device).view(-1, 1, 1)
        return (input - mean) / std


class ResizeLayer(torch.nn.Module):
    def __init__(self, out_size):
        super(ResizeLayer, self).__init__()
        self.transform = transforms.Compose([
            transforms.Resize(out_size, interpolation=3)
        ])
    def forward(self, input: torch.tensor):
        return self.transform(input)


class ResizeWrapper(torch.nn.Module):
    def __init__(self, model, in_size, out_size):
        super(ResizeWrapper, self).__init__()
        self.model = model
        self.resize_in = transforms.Resize(in_size, interpolation=3)
        self.resize_out = transforms.Resize(out_size, interpolation=3)

    def forward(self, x, *args, **kwargs):
        x = self.resize_in(x)
        x = self.model(x, *args, **kwargs)
        out = self.resize_out(x)
        return out


class CropWrapper(torch.nn.Module):
    def __init__(self, model, in_size, out_size):
        super(CropWrapper, self).__init__()
        self.model = model
        pad = int((in_size - out_size) // 2)
        self.padding = torch.nn.ReflectionPad2d(pad)
        self.crop = transforms.CenterCrop(out_size)

    def forward(self, x, *args, **kwargs):
        x = self.padding(x)
        x = self.model(x, *args, **kwargs)
        out = self.crop(x)
        return out


class IDRSCIFAR10(datasets.CIFAR10):
    def __init__(self, root, sigma_path, *args, **kwargs):
        super().__init__(root, *args, **kwargs)
        self.sigma_path = sigma_path

        indices, sigmas = [], []

        df = pd.read_csv(sigma_path, delimiter="\t")
        x = df['idx'].tolist()
        s = df['sigma'].tolist()
        indices.extend(x)
        sigmas.extend(s)

        self._indices = indices
        self._sigmas = sigmas

    def __getitem__(self, index: int):
        idx = self._indices[index]
        sigma = self._sigmas[idx]

        img, target = super().__getitem__(idx)
        return img, target, sigma

    def __len__(self):
        return len(self._indices)


def get_dataset(args, dataset):
    """Return the dataset as a PyTorch Dataset object"""
    mean = IMAGENET_INCEPTION_MEAN
    std = IMAGENET_INCEPTION_STD
    normalize = NormalizeLayer(mean, std)

    if dataset == "imagenet":
        im_size = 224
        n_classes = 1000
    elif dataset in ["imagenet_a", "imagenet_r"]:
        im_size = 224
        n_classes = 1000
    elif dataset == "cifar10":
        im_size = 32
        n_classes = 10
    elif dataset == "cifar10_train":
        im_size = 32
        n_classes = 10
    elif dataset == "cifar10.1":
        im_size = 32
        n_classes = 10
    elif "cifar10c" in dataset:
        im_size = 32
        n_classes = 10
    elif dataset == "cifar10_idrs":
        im_size = 32
        n_classes = 10
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(im_size, interpolation=3),
        transforms.CenterCrop(im_size),
        normalize
    ])
    if dataset == "imagenet":
        val_dir = os.path.join(args.data_path, 'ImageNet/val/')
        ds = datasets.ImageFolder(val_dir, transform)
    elif dataset == "imagenet_a":
        val_dir = os.path.join(args.data_path, 'imagenet-a')
        ds = datasets.ImageFolder(val_dir, transform)
    elif dataset == "imagenet_r":
        val_dir = os.path.join(args.data_path, 'imagenet-r')
        ds = datasets.ImageFolder(val_dir, transform)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(args.data_path, train=False, transform=transform, download=True)
    elif dataset == "cifar10.1":
        data_path = os.path.join(args.data_path, 'CIFAR-10.1/datasets')

        test_images = np.load(os.path.join(data_path, 'cifar10.1_v6_data.npy'))
        test_images = np.transpose(test_images, (0, 3, 1, 2))
        test_images = torch.from_numpy(test_images)
        test_labels = torch.from_numpy(np.load(os.path.join(data_path, 'cifar10.1_v6_labels.npy')))
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(32),
            transforms.ToTensor()
        ])
        ds = TransformTensorDataset([test_images, test_labels], transform=transform)
    elif dataset == "cifar10_train":
        ds = datasets.CIFAR10(args.data_path, train=True, transform=transform, download=True)
    elif "cifar10c" in dataset:
        corruption = dataset.replace('cifar10c_', '')
        ds = load_cifar10c(args.data_path, corruption, transform=transform)
    elif dataset == "cifar10_idrs":
        path = "OUTPUT/idrs_sigma/cifar10_r0.01.tsv"
        ds = IDRSCIFAR10(args.data_path, path, train=False, transform=transform, download=True)
    else:
        raise NotImplementedError()
    return ds, n_classes


def get_diffusion_model(dataset, model_path=None):
    if dataset == "cifar10" or ("cifar10c" in dataset) or dataset == "cifar10.1" or dataset == 'cifar10_idrs':
        from improved_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion
        )
        model_args = model_and_diffusion_defaults()
        args = {
            "image_size": 32,
            "num_channels": 128,
            "num_res_blocks": 3,
            "learn_sigma": True,
            "dropout": 0.3,
            "diffusion_steps": 4000,
            "noise_schedule": "cosine"
        }
        if model_path is None:
            model_path = 'OUTPUT/denoising_models/cifar10_uncond_50M_500K.pt'
    elif "imagenet" in dataset:
        from guided_diffusion.script_util import (
            model_and_diffusion_defaults,
            create_model_and_diffusion
        )
        model_args = model_and_diffusion_defaults()
        args = {
            "attention_resolutions": "32,16,8",
            "image_size": 256,
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "learn_sigma": True,
            "diffusion_steps": 1000,
            "noise_schedule": "linear",
            "use_fp16": True,
            "use_scale_shift_norm": True
        }
        if model_path is None:
            model_path = 'OUTPUT/denoising_models/imagenet/256x256_diffusion_uncond.pt'
    else:
        raise NotImplementedError()
    model_args.update(args)

    model, diffusion = create_model_and_diffusion(**model_args)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)

    if "imagenet" in dataset:
        model = CropWrapper(model, 256, 224)

    model.cuda()
    return model, diffusion