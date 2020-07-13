#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import torch
import torch.nn as nn
from torch.optim import LBFGS, SGD
from torch.optim.lr_scheduler import StepLR

import dataloading
import modules
import resnet
import util

MODULE_TYPES = ["Linear", "Conv2d", "GroupNorm"]


def add_l2_regularization(criterion, model, regularization_param):
    """
    Adds an L2-regularizer on the parameters in the `model` to the loss function
    `criterion`. The regularization parameter is given by `regularization_param`.
    """

    def regularized_loss(predictions, targets):
        loss = criterion(predictions, targets)
        for param in model.parameters():
            loss += (regularization_param / 2.) * param.flatten().dot(param.flatten())
        return loss

    return regularized_loss


def initialize_model(num_inputs, num_outputs, model="linear", device="cpu"):
    """
    Initializes linear model with specified number of inputs and outputs.
    """

    # load model:
    model_name = model
    if model_name == "linear":
        model = nn.Linear(num_inputs, num_outputs)
    elif model_name.startswith("resnet"):

        # get a vanilla ResNet model:
        assert hasattr(resnet, model_name), f"Unknown model: {model_name}"
        model = getattr(resnet, model_name)()
        # TODO: Add checks that number of inputs and outputs match.

        # replace all batchnorm layers by groupnorm layers:
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):

                # create groupnorm layer:
                new_module = nn.GroupNorm(
                    min(32, module.num_features),
                    module.num_features,
                    affine=(module.weight is not None and module.bias is not None),
                )

                # replace the layer:
                parent = model
                name_list = name.split(".")
                for name in name_list[:-1]:
                    parent = parent._modules[name]
                parent._modules[name_list[-1]] = new_module

    else:
        raise ValueError(f"Unknown model: {model_name}")

    # copy model to GPU(s) and return:
    if device == "gpu":
        assert torch.cuda.is_available(), "CUDA is not available on this machine."
        logging.info("Copying model to GPU...")
        model.cuda()
    return model


def privatize_model(model, clip, std):
    """
    Converts a "normal" model into a model that computes private gradients.
    """
    types = tuple(getattr(nn, mod_type) for mod_type in MODULE_TYPES)
    private_types = tuple(getattr(modules, mod_type) for mod_type in MODULE_TYPES)
    for module in model.modules():
        if isinstance(module, types) and not isinstance(module, private_types):
            typename = str(type(module))
            typename = typename[typename.rfind(".") + 1:-2]
            module.__class__ = getattr(modules, typename)
            module.clip = torch.tensor(clip)
            module.std = torch.tensor(std)
        else:
            if hasattr(module, "weight") or hasattr(module, "bias"):
                raise NotImplementedError(
                    f"Privacy conversion of {type(module)} not implemented."
                )
    return model


def unprivatize_model(model):
    """
    Converts a model that computes private gradients into a "normal" model.
    """
    types = tuple(getattr(modules, mod_type) for mod_type in MODULE_TYPES)
    for module in model.modules():
        if isinstance(module, types):
            typename = str(type(module))
            typename = typename[typename.rfind(".") + 1:-2]
            module.__class__ = getattr(nn, typename)
            del module.clip
            del module.std
    return model


def train_model(model, dataset, optimizer="lbfgs", batch_size=128, num_epochs=100,
                learning_rate=1., criterion=None, augmentation=False, momentum=0.9,
                use_lr_scheduler=True, visualizer=None, title=None):
    """
    Trains `model` on samples from the specified `dataset` using the specified
    `optimizer` ("lbfgs" or "sgd") with batch size `batch_size` for `num_epochs`
    epochs to minimize the specified `criterion` (default = `nn.CrossEntropyLoss`).

    For L-BFGS, the batch size is ignored and full gradients are used. The
    `learning_rate` is only used as initial value; step sizes are determined by
    checking the Wolfe conditions.

    For SGD, the initial learning rate is set to `learning_rate` and is reduced
    by a factor of 10 four times during training. Training uses Nesterov momentum
    of 0.9. Optionally, data `augmentation` can be enabled as well.

    Training progress is shown in the visdom `visualizer` in a window with the
    specified `title`.
    """

    # set up optimizer, criterion, and learning curve:
    model.train()
    device = next(model.parameters()).device
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if visualizer is not None:
        window = [None]

    # set up optimizer and learning rate scheduler:
    if optimizer == "sgd":
        optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        scheduler = StepLR(optimizer, step_size=max(1, num_epochs // 4), gamma=0.1)
    elif optimizer == "lbfgs":
        assert not augmentation, "Cannot use data augmentation with L-BFGS."
        use_lr_scheduler = False
        optimizer = LBFGS(
            model.parameters(),
            lr=learning_rate,
            tolerance_grad=1e-4,
            line_search_fn="strong_wolfe",
        )
        batch_size = len(dataset["targets"])
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")

    # create data sampler:
    transform = dataloading.data_augmentation() if augmentation else None
    datasampler = dataloading.load_datasampler(
        dataset, batch_size=batch_size, transform=transform
    )

    # perform training epochs:
    for epoch in range(num_epochs):
        num_samples, total_loss = 0, 0.
        for sample in datasampler():

            # copy sample to correct device if needed:
            for key in sample.keys():
                if sample[key].device != device:
                    sample[key] = sample[key].to(device=device)

            # closure that performs forward-backward pass:
            def loss_closure():
                optimizer.zero_grad()
                predictions = model(sample["features"])
                loss = criterion(predictions, sample["targets"])
                loss.backward()
                return loss

            # perform parameter update:
            loss = optimizer.step(closure=loss_closure)

            # aggregate loss values for monitoring:
            total_loss += (loss.item() * sample["features"].size(0))
            num_samples += sample["features"].size(0)

        # decay learning rate (SGD only):
        if use_lr_scheduler and epoch != num_epochs - 1:
            scheduler.step()

        # print statistics:
        if epoch % 10 == 0:
            average_loss = total_loss / float(num_samples)
            logging.info(f" => epoch {epoch + 1}: loss = {average_loss}")
            if visualizer is not None:
                window[0] = util.learning_curve(
                    visualizer,
                    torch.LongTensor([epoch + 1]),
                    torch.DoubleTensor([average_loss]),
                    window=window[0],
                    title=title,
                )

    # we are done training:
    model.eval()


def test_model(model, dataset, batch_size=128, augmentation=False):
    """
    Evaluates `model` on samples from the specified `dataset` using the specified
    `batch_size`. Returns predictions for all samples in the dataset. Optionally,
    test-time data `augmentation` can be enabled as well.
    """

    # create data sampler:
    model.eval()
    device = next(model.parameters()).device
    transform = dataloading.data_augmentation(train=False) if augmentation else None
    datasampler = dataloading.load_datasampler(
        dataset, batch_size=batch_size, transform=transform, shuffle=False
    )

    # perform test pass:
    predictions = []
    for sample in datasampler():

        # copy sample to correct device if needed:
        for key in sample.keys():
            if sample[key].device != device:
                sample[key] = sample[key].to(device=device)

        # make predictions:
        with torch.no_grad():
            predictions.append(model(sample["features"]))

    # return all predictions:
    return torch.cat(predictions, dim=0)


def get_parameter_vector(model):
    """
    Returns all parameters in the specified `model` in a single vector.

    Alternatively, `model` can also be an iterable of parameters.
    """
    if isinstance(model, nn.Module):
        return torch.nn.utils.parameters_to_vector(model.parameters())
    elif hasattr(model, "__iter__"):
        return torch.nn.utils.parameters_to_vector(model)
    else:
        raise ValueError("Model is not nn.Module or iterable.")


def set_parameter_vector(model, parameters):
    """
    Sets parameters in the specified `model` to values in `parameters` vector.

    Alternatively, `model` can also be an iterable of parameters.
    """
    if isinstance(model, nn.Module):
        torch.nn.utils.vector_to_parameters(parameters, model.parameters())
    elif hasattr(model, "__iter__"):
        torch.nn.utils.vector_to_parameters(parameters, model)
    else:
        raise ValueError("Model is not nn.Module or iterable.")
