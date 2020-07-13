#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearFunction(torch.autograd.Function):
    """
    Linear function that computes gradients in a differentially private way.

    This uses https://arxiv.org/abs/1510.01799 to clip the per-example gradients
    without explicitly constructing them.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, clip, std):

        # save context for backward:
        ctx.save_for_backward(input, weight, bias)
        ctx.clip = clip
        ctx.std = std

        # apply linear function and return:
        output = input.mm(weight.t())
        if bias is not None:
            output.add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        # unpack saved context:
        input, weight, bias = ctx.saved_tensors

        # compute gradient with respect to input (not private):
        grad_input = grad_output.mm(weight)

        # compute norms of per-example gradients with respect to parameters:
        gradient_norm = grad_output.pow(2.).sum(1, keepdim=True).mul(
            input.pow(2.).sum(1, keepdim=True)
        )
        gradient_norm = torch.sqrt(gradient_norm)

        # aggregate the clipped per-example gradients:
        multiplier = _get_multipliers(gradient_norm, ctx.clip)
        grad_weight = grad_output.div(multiplier + 1e-9).t().matmul(input)

        # add noise to gradient:
        grad_weight += torch.randn_like(grad_weight) * ctx.clip * ctx.std

        # perform same procedure for bias gradients:
        if bias is not None:
            multiplier = _get_multipliers(grad_output.norm(2, 1), ctx.clip)
            grad_bias = grad_output.mul(multiplier.unsqueeze(1)).sum(0)
            grad_bias += torch.randn_like(grad_bias) * ctx.clip * ctx.std
        else:
            grad_bias = None

        # return private gradients:
        return grad_input, grad_weight, grad_bias, None, None


class Conv2dFunction(torch.autograd.Function):
    """
    Conv2d function that computes gradients in a differentially private way.
    """

    @staticmethod
    def forward(ctx, input, weight, stride, padding, clip, std):

        # save context for backward:
        ctx.save_for_backward(input, weight)
        ctx.stride = stride
        ctx.padding = padding
        ctx.clip = clip
        ctx.std = std

        # perform convolution:
        return F.conv2d(
            input,
            weight,
            bias=None,
            stride=stride,
            padding=padding,
            dilation=1,
            groups=1,
        )

    @staticmethod
    def backward(ctx, grad_output):

        # get input, kernel, and sizes:
        input, weight = ctx.saved_tensors
        batch_size = input.size(0)
        out_channels, in_channels, weight_size_y, weight_size_x = weight.size()
        assert input.size(1) == in_channels, "wrong number of input channels"
        assert grad_output.size(1) == out_channels, "wrong number of output channels"
        assert grad_output.size(0) == batch_size, "wrong batch size"

        # compute gradient with respect to input:
        grad_input = torch.nn.grad.conv2d_input(
            input.size(),
            weight,
            grad_output,
            stride=ctx.stride,
            padding=ctx.padding,
        )

        # compute per-example gradient with respect to weights:
        out_channels, in_channels, weight_size_y, weight_size_x = weight.size()
        grad_output = grad_output.contiguous().repeat(1, in_channels, 1, 1)
        grad_output = grad_output.contiguous().view(
            grad_output.shape[0] * grad_output.shape[1],
            1,
            grad_output.shape[2],
            grad_output.shape[3],
        )
        input = input.contiguous().view(
            1,
            input.shape[0] * input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        grad_weight = torch.conv2d(
            input,
            grad_output,
            None,
            1,
            ctx.padding,
            ctx.stride,
            in_channels * batch_size,
        )
        grad_weight = grad_weight.contiguous().view(
            batch_size,
            grad_weight.shape[1] // batch_size,
            grad_weight.shape[2],
            grad_weight.shape[3],
        )

        # compute norm of per-example weight gradients:
        grad_norm = torch.norm(
            grad_weight.view(batch_size, -1), p='fro', dim=1, keepdim=True,
        ).view(batch_size, 1, 1, 1)

        # aggregate the clipped per-example weight gradients:
        multiplier = _get_multipliers(grad_norm, ctx.clip)
        grad_weight = grad_weight.mul_(multiplier).sum(dim=0)
        grad_weight = grad_weight.view(
            in_channels,
            out_channels,
            grad_weight.shape[1],
            grad_weight.shape[2]
        ).transpose(0, 1).narrow(2, 0, weight_size_y).narrow(3, 0, weight_size_x)

        # add noise to gradient:
        grad_weight += torch.randn_like(grad_weight) * ctx.clip * ctx.std

        # return gradients:
        return grad_input, grad_weight, None, None, None, None


class AffineFunction(torch.autograd.Function):
    """
    Affine function that computes gradients in a differentially private way.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, clip, std):

        # save context for backward pass:
        ctx.save_for_backward(input, weight, bias)
        ctx.clip = clip
        ctx.std = std

        # shape for broadcasting weights with input:
        broadcast_shape = [1] * input.dim()
        broadcast_shape[1] = input.shape[1]
        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        # apply affine transform to output before returning:
        output = input.mul(weight).add_(bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        # unpack context from forward pass:
        input, weight, bias = ctx.saved_tensors

        # determine dimensions over which affine transform is broadcasted:
        stats_dimensions = list(range(grad_output.dim()))
        stats_dimensions.pop(1)

        # shape for broadcasting weights with grad_output:
        broadcast_shape = [1] * input.dim()
        broadcast_shape[1] = input.shape[1]
        weight = weight.reshape(broadcast_shape)
        bias = bias.reshape(broadcast_shape)

        # compute gradient with respect to input:
        grad_input = grad_output.mul(weight)

        # compute per-example gradient with respect to weights and biases:
        grad_weight = grad_output.mul(input)
        grad_bias = grad_output.clone()

        # compute norms of per-example gradients:
        batch_size = grad_output.size(0)
        grad_weight_norm = torch.norm(
            grad_weight.view(batch_size, -1), p='fro', dim=1, keepdim=True,
        )
        grad_bias_norm = torch.norm(
            grad_bias.view(batch_size, -1), p='fro', dim=1, keepdim=True,
        )

        # shape for broadcasting multipliers with grad_output:
        broadcast_shape = [1] * grad_output.dim()
        broadcast_shape[0] = grad_output.size(0)

        # aggregate the clipped per-example weight gradients:
        multiplier = _get_multipliers(grad_weight_norm, ctx.clip)
        multiplier = multiplier.reshape(broadcast_shape)
        grad_weight = grad_weight.mul_(multiplier).sum(stats_dimensions)

        # aggregate the clipped per-example weight gradients:
        multiplier = _get_multipliers(grad_bias_norm, ctx.clip)
        multiplier = multiplier.reshape(broadcast_shape)
        grad_bias = grad_bias.mul_(multiplier).sum(stats_dimensions)

        # add noise to gradients:
        grad_weight += torch.randn_like(grad_weight) * ctx.clip * ctx.std
        grad_bias += torch.randn_like(grad_bias) * ctx.clip * ctx.std

        # return gradients:
        return grad_input, grad_weight, grad_bias, None, None


# alias the application of the custom AutogradFunctions:
linear = LinearFunction.apply
conv2d = Conv2dFunction.apply
affine = AffineFunction.apply


class Linear(nn.Linear):
    """
    Extends nn.Linear to compute gradients in a differentially private way.
    Please refer to the documentation of nn.Linear for constructor signature.
    """

    def __init__(self, in_channels, out_channels, bias=True, clip=math.inf, std=0.0):
        super(Linear, self).__init__(in_channels, out_channels, bias=bias)
        self.clip = clip
        self.std = std

    def forward(self, input):
        if self.train:
            return linear(input, self.weight, self.bias, self.clip, self.std)
        else:
            return F.linear(input, self.weight, self.bias)


class Conv2d(nn.Conv2d):
    """
    Extends nn.Linear to compute gradients in a differentially private way.
    Please refer to the documentation of nn.Conv2d for constructor signature.
    This module does not support group convolution, dilation, or biases.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        clip=math.inf,
        std=0.0,
    ):
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.clip = clip
        self.std = std
        assert self.bias is None, "Does not support bias in convolutions."
        assert self.groups == 1, "Does not support group convolutions."
        assert self.dilation == 1, "Does not support dilated convolutions."

    def forward(self, input):
        if self.train:
            return conv2d(
                input, self.weight, self.stride, self.padding, self.clip, self.std
            )
        else:
            return F.conv2d(
                input,
                self.weight,
                bias=self.bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )


class GroupNorm(nn.GroupNorm):
    """
    Extends nn.GroupNorm to compute gradients in a differentially private way.
    Please refer to the documentation of nn.GroupNorm for constructor signature.
    """

    def __init__(
        self, num_groups, num_channels, eps=1e-5, affine=True, clip=math.inf, std=0.0
    ):
        super(GroupNorm, self).__init__(
            num_groups, num_channels, eps=eps, affine=affine
        )
        self.clip = clip
        self.std = std

    def forward(self, input):

        # training mode:
        if self.train:
            input_norm = F.group_norm(
                input, self.num_groups, None, None, self.eps
            )
            if self.weight is None or self.bias is None:
                return input_norm
            else:
                return affine(
                    input_norm, self.weight, self.bias, self.clip, self.std
                )  # only the affine transform needs private gradients

        # inference mode:
        else:
            return F.group_norm(
                input, self.num_groups, self.weight, self.bias, self.eps
            )


def _get_multipliers(gradient_norm, clip):
    """
    Given a vector of per-example `gradient_norm`s, computes the multipliers to
    be applied to the gradients to clip them at value `clip`.
    """
    multiplier = gradient_norm.new(gradient_norm.size()).fill_(1.)
    multiplier[gradient_norm.gt(clip)] = clip / gradient_norm[gradient_norm.gt(clip)]
    return multiplier
