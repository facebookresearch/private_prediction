#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import torch
from torchvision.datasets import MNIST, CIFAR10, CIFAR100
import torchvision.transforms as transforms
from torchvision.datasets.mnist import read_image_file, read_label_file


class MNIST1M(MNIST):
    """
    MNIST1M dataset that can be generated using InfiMNIST.

    Note: This dataset cannot be downloaded automatically.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        super(MNIST1M, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def download(self):
        """
        Process MNIST1M data if it does not exist in processed_folder already.
        """

        # check if processed data does not exist:
        if self._check_exists():
            return

        # process and save as torch files:
        logging.info("Processing MNIST1M data...")
        os.makedirs(self.processed_folder, exist_ok=True)
        training_set = (
            read_image_file(os.path.join(self.raw_folder, "mnist1m-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "mnist1m-labels-idx1-ubyte"))
        )
        test_set = (
            read_image_file(os.path.join(self.raw_folder, "t10k-images-idx3-ubyte")),
            read_label_file(os.path.join(self.raw_folder, "t10k-labels-idx1-ubyte"))
        )
        with open(os.path.join(self.processed_folder, self.training_file), "wb") as f:
            torch.save(training_set, f)
        with open(os.path.join(self.processed_folder, self.test_file), "wb") as f:
            torch.save(test_set, f)
        logging.info("Done!")


def load_dataset(
    name="mnist",
    split="train",
    normalize=True,
    reshape=True,
    num_classes=None,
    root="/tmp",
):
    """
    Loads train or test `split` from the dataset with the specified `name`
    (mnist, cifar10, or cifar 100). Setting `normalize` to `True` (default)
    normalizes each feature vector to lie on the unit ball. Setting `reshape`
    to `True` (default) flattens the images into feature vectors. Specifying
    `num_classes` selects only the first `num_classes` of the classification
    problem (default: all classes).
    """

    # assertions:
    assert split in ["train", "test"], f"unknown split: {split}"
    datasets = {
        "mnist": MNIST,
        "mnist1m": MNIST1M,
        "cifar10": CIFAR10,
        "cifar100": CIFAR100,
    }
    assert name in datasets, f"unknown dataset: {name}"

    # download the image dataset:
    dataset = datasets[name](
        f"{root}/{name}_original",
        download=True,
        train=(split == "train"),
    )

    # preprocess the image dataset:
    features, targets = dataset.data, dataset.targets
    if not torch.is_tensor(features):
        features = torch.from_numpy(features)
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    features = features.float().div_(255.)
    targets = targets.long()

    # flatten images or convert to NCHW:
    if reshape:
        features = features.reshape(features.size(0), -1)
    else:
        features = features.transpose(1, 3).transpose(2, 3)

    # select only subset of classes:
    if num_classes is not None:
        assert num_classes >= 2, "number of classes must be >= 2"
        mask = targets.lt(num_classes)  # assumes classes are 0, 1, ..., C - 1
        features = features[mask, :]
        targets = targets[mask]

    # normalize all samples to lie within unit ball:
    if normalize:
        assert reshape, "normalization without reshaping unsupported"
        features.div_(features.norm(dim=1).max())

    # return dataset:
    return {"features": features, "targets": targets}


def load_datasampler(dataset, batch_size=1, shuffle=True, transform=None):
    """
    Returns a data sampler that yields samples of the specified `dataset` with the
    given `batch_size`. An optional `transform` for samples can also be given.
    If `shuffle` is `True` (default), samples are shuffled.
    """
    assert dataset["features"].size(0) == dataset["targets"].size(0), \
        "number of feature vectors and targets must match"
    if transform is not None:
        assert callable(transform), "transform must be callable if specified"
    N = dataset["features"].size(0)

    # define simple dataset sampler:
    def sampler():
        idx = 0
        perm = torch.randperm(N) if shuffle else torch.range(0, N).long()
        while idx < N:

            # get batch:
            start = idx
            end = min(idx + batch_size, N)
            batch = dataset["features"][perm[start:end], :]

            # apply transform:
            if transform is not None:
                transformed_batch = [
                    transform(batch[n, :]) for n in range(batch.size(0))
                ]
                batch = torch.stack(transformed_batch, dim=0)

            # return sample:
            yield {"features": batch, "targets": dataset["targets"][perm[start:end]]}
            idx += batch_size

    # return sampler:
    return sampler


def data_augmentation(train=True):
    """
    Returns function that performs data augmentation on samples. If `train` is
    set to `False`, returns the corresponding normalizing transform.
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        normalize,
    ])
    return transform if train else normalize


def subsample(data, num_samples, random=True):
    """
    Subsamples the specified `data` to contain `num_samples` samples. Set
    `random` to `False` to not select samples randomly but only pick top ones.
    """

    # assertions:
    assert isinstance(data, dict), "data must be a dict"
    assert "targets" in data, "data dict does not have targets field"
    dataset_size = data["targets"].nelement()
    assert num_samples > 0, "num_samples must be positive integer value"
    assert num_samples <= dataset_size, "num_samples cannot exceed data size"

    # subsample data:
    if random:
        permutation = torch.randperm(dataset_size)
    for key, value in data.items():
        if random:
            data[key] = value.index_select(0, permutation[:num_samples])
        else:
            data[key] = value.narrow(0, 0, num_samples).contiguous()
    return data


def pca(data, num_dims=None, mapping=None):
    """
    Applies PCA on the specified `data` to reduce its dimensionality to
    `num_dims` dimensions, and returns the reduced data and `mapping`.

    If a `mapping` is specified as input, `num_dims` is ignored and that mapping
    is applied on the input `data`.
    """

    # work on both data tensor and data dict:
    data_dict = False
    if isinstance(data, dict):
        assert "features" in data, "data dict does not have features field"
        data_dict = True
        original_data = data
        data = original_data["features"]
    assert data.dim() == 2, "data tensor must be two-dimensional matrix"

    # compute PCA mapping:
    if mapping is None:
        assert num_dims is not None, "must specify num_dims or mapping"
        mean = torch.mean(data, 0, keepdim=True)
        zero_mean_data = data.sub(mean)
        covariance = torch.matmul(zero_mean_data.t(), zero_mean_data)
        _, projection = torch.symeig(covariance, eigenvectors=True)
        projection = projection[:, -min(num_dims, projection.size(1)):]
        mapping = {"mean": mean, "projection": projection}
    else:
        assert isinstance(mapping, dict), "mapping must be a dict"
        assert "mean" in mapping and "projection" in mapping, "mapping missing keys"
        if num_dims is not None:
            logging.warning("Value of num_dims is ignored when mapping is specified.")

    # apply PCA mapping:
    reduced_data = data.sub(mapping["mean"]).matmul(mapping["projection"])

    # return results:
    if data_dict:
        original_data["features"] = reduced_data
        reduced_data = original_data
    return reduced_data, mapping
