#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import torch


def learning_curve(visualizer, idx, value, window=None, dtype=torch.double, title=""):
    """
    Appends new value to learning curve, creating new curve if none exists.
    """
    opts = {
        "title": title,
        "xlabel": "Epoch",
        "ylabel": "Loss value",
    }
    window = visualizer.line(value.view(value.nelement(), 1), idx,
                             update=None if window is None else "append",
                             opts=opts,
                             win=window,
                             env="shared-model")
    return window


def binary_accuracy(predictions, targets):
    """
    Measures accuracy of `predictions` for binary classification by comparing
    them with given `targets`.
    """
    accuracy = ((predictions > 0.) == (targets == 1)).sum() / float(targets.nelement())
    return accuracy


def accuracy(predictions, targets):
    """
    Measures accuracy of predictions for multi-way classification. Inputs are
    expected to be two matrices with rows containing one-hot or probability
    vectors of length K, `vector1` and `vector2`.
    """
    _, predictions = predictions.topk(1, dim=1, largest=True, sorted=True)
    _, targets = targets.topk(1, dim=1, largest=True, sorted=True)
    accuracy = predictions.eq(targets).sum() / float(targets.size(0))
    return accuracy


def binary_search(func, constraint, minimum, maximum, tol=1e-5):
    """
    Performs binary search on monotonically increasing function `func` between
    `minimum` and `maximum` to find the maximum value for which the function's
    output satisfies the specified `constraint` (which is a binary function).
    Returns maximum value `x` at which `constraint(func(x))` is `True`.

    The function takes an optional parameter specifying the tolerance `tol`.
    """
    assert constraint(func(minimum)), "constraint on function must hold at minimum"

    # evaluate function at maximum:
    if constraint(func(maximum)):
        return maximum

    # perform the binary search:
    while maximum - minimum > tol:
        midpoint = (minimum + maximum) / 2.
        if constraint(func(midpoint)):
            minimum = midpoint
        else:
            maximum = midpoint

    # return value:
    return minimum
