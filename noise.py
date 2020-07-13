#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch


def categorical(probabilities, device=None):
    """
    Sample from a categorical distribution with specified `probabilities`.
    """
    sample = torch.distributions.categorical.Categorical(probs=probabilities).sample()
    if device is not None:
        sample = sample.to(device=device)
    return sample


def gaussian(mean, precision, device=None):
    """
    Samples from a Gaussian distribution with the specified `mean` and `precision`.
    """
    sample = torch.distributions.normal.Normal(mean, 1. / precision).sample()
    if device is not None:
        sample = sample.to(device=device)
    return sample


def laplacian(mean, precision, device=None):
    """
    Samples from a Laplace distribution with the specified `mean` and `precision`.
    """
    sample = torch.distributions.laplace.Laplace(mean, 1. / precision).sample()
    if device is not None:
        sample = sample.to(device=device)
    return sample


def sqrt_gaussian(mean, precision, device=None):
    """
    Samples from a square-root Gaussian distribution with the specified
    DxN_1xN_2x...xN_n  tensor `mean` and tensor `precision` which should be a
    scalar or have shape N_1xN_2x...xN_n. The zero-mean square-root Gaussian
    distribution is a multi-variate distribution probability density function:

    p(x) \propto exp(-precision * ||x||_2) with x \in \mathbb{R}^D

    Each column in the output corresponds to a single D-dimensional sample.
    """  # noqa: W905
    assert isinstance(precision, float) or precision.nelement() == 1 \
        or precision.shape == mean.shape[1:], "Invalid shape for precision."

    # sample directions as unit-norm Gaussian vectors:
    direction = torch.randn(mean.size(), device=mean.device)  # samples from Gaussian
    norm = torch.norm(direction, dim=0, keepdim=True)         # norm of the samples
    direction.div_(norm)                                      # unit-norm samples

    # sample norms from Gamma distribution:
    shape = torch.ones(mean.shape[1:]).mul_(mean.size(0))
    norm = torch.distributions.gamma.Gamma(shape, precision).sample()
    if norm.device != direction.device:
        norm = norm.to(direction.device)

    # return final sample:
    sample = direction.mul_(norm).add_(mean)
    if device is not None:
        sample = sample.to(device=device)
    return sample
