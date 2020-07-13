#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import json
import logging
import math
import os
import torch
import visdom

import dataloading
import private_prediction

# set up logger:
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# hyperparameter values to try in cross-validation:
CROSS_VALIDATION = {
    "weight_decay": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1],
    "clip": [5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1],
}


def cross_validate(args, data, visualizer=None):
    """
    Performs cross-validation over hyperparameters for which this was requested.
    """

    # check if there are any parameters to cross-validate over:
    accuracies = {}
    arguments = {key: getattr(args, key) for key in CROSS_VALIDATION.keys()}
    if any(value == -1 for value in arguments.values()):

        # create validation split from training data:
        valid_size = data["train"]["features"].size(0) // 10
        original_train, data["valid"] = {}, {}
        for key in data["train"].keys():
            original_train[key] = data["train"][key]
            data["valid"][key] = original_train[key].narrow(0, 0, valid_size)
            data["train"][key] = original_train[key].narrow(
                0, valid_size, original_train[key].size(0) - valid_size
            )
        # NOTE: This assumes data is already shuffled.
        # NOTE: This makes an additional data copy, which may be bad on GPUs.

        # get hyperparameter key and values:
        hyper_key = [key for key, val in arguments.items() if val == -1]
        assert len(hyper_key) == 1, \
            "can only cross-validate over single hyperparameter at the same time"
        hyper_key = hyper_key[0]
        hyper_values = CROSS_VALIDATION[hyper_key]

        # perform the actual cross-validation:
        num_repetitions, idx = max(1, args.num_repetitions // 10), 0
        for hyper_value in hyper_values:

            # make copy of arguments that we can alter:
            args_copy = copy.deepcopy(args)
            setattr(args_copy, hyper_key, hyper_value)
            if args_copy.inference_budget == -1:
                args_copy.inference_budget = 100
            accuracies[hyper_value] = {}

            # repeat experiment multiple times:
            for _ in range(num_repetitions):
                logging.info(f"Cross-validation experiment {idx + 1} of "
                             f"{len(hyper_values) * num_repetitions}...")
                private_prediction.compute_accuracy(
                    args_copy, data,
                    accuracies=accuracies[hyper_value],
                    visualizer=visualizer,
                )
                idx += 1

        # find best hyperparameter setting:
        for hyper_value in hyper_values:
            valid_accuracy = accuracies[hyper_value]["valid"]
            if isinstance(valid_accuracy, dict):  # inference budget in accuracies
                valid_accuracy = valid_accuracy[str(args_copy.inference_budget)]
            accuracies[hyper_value] = sum(valid_accuracy) / float(num_repetitions)
        optimal_value = max(accuracies, key=accuracies.get)
        logging.info(f"Selecting {hyper_key} value of {optimal_value}...")

        # clean up validation set:
        data["train"] = original_train
        del data["valid"]

        # update arguments object:
        setattr(args, hyper_key, optimal_value)

    # return arguments to use for main experiment:
    return args


def main(args):
    """
    Runs private predictions experiment on dataset using input arguments `args`.
    """

    # set up visualizer:
    if args.visdom:
        visualizer = visdom.Visdom(args.visdom)
    if not args.visdom or not visualizer.check_connection():
        visualizer = None

    # load dataset:
    logging.info(f"Loading {args.dataset} dataset...")
    normalize = args.dataset.startswith("mnist")
    reshape = (args.model == "linear")
    num_classes = None if args.num_classes == -1 else args.num_classes
    data = {}
    for split in ["train", "test"]:
        data[split] = dataloading.load_dataset(
            name=args.dataset,
            split=split,
            normalize=normalize,
            reshape=reshape,
            num_classes=num_classes,
            root=args.data_folder,
        )

    # apply PCA if requested (on all data; non-transductive setting):
    if args.pca_dims != -1:
        assert reshape, "cannot use PCA with non-linear models"
        data["train"], mapping = dataloading.pca(data["train"], num_dims=args.pca_dims)
        data["test"], _ = dataloading.pca(data["test"], mapping=mapping)

    # subsample training data if requested:
    if args.num_samples != -1:
        data["train"] = dataloading.subsample(
            data["train"], num_samples=args.num_samples, random=False,
        )

    # copy data to GPU if requested (for linear models only):
    if args.device == "gpu" and args.model == "linear":
        assert torch.cuda.is_available(), "CUDA is not available on this machine."
        logging.info("Copying data to GPU...")
        for split in data.keys():
            for key, value in data[split].items():
                data[split][key] = value.cuda()

    # use cross-validation to tune hyperparameters:
    args = cross_validate(args, data, visualizer=visualizer)

    # repeat the same experiment multiple times:
    accuracies = {}
    for idx in range(args.num_repetitions):
        logging.info(f"Experiment {idx + 1} of {args.num_repetitions}...")
        private_prediction.compute_accuracy(
            args, data, accuracies=accuracies, visualizer=visualizer
        )

    # save results to file:
    if args.result_file is not None and args.result_file != "":
        logging.info(f"Writing results to file {args.result_file}...")
        with open(args.result_file, "wt") as json_file:
            json.dump(accuracies, json_file)


# run all the things:
if __name__ == '__main__':

    # parse input arguments:
    hostname = os.environ.get("HOSTNAME", "localhost")
    parser = argparse.ArgumentParser(description="Private prediction")
    parser.add_argument("--dataset", default="mnist", type=str,
                        help="dataset: 'mnist' (default), 'mnist1m', 'cifar10', "
                        "or 'cifar100'")
    parser.add_argument("--model", default="linear", type=str,
                        help="model: 'linear' (default) or "
                        "'resnet{20,32,44,56,110,1202}'")
    parser.add_argument("--method", default="subsagg", type=str,
                        help="private prediction method: 'subsagg' (default), "
                        "'loss_perturbation', '{model,logit}_sensitivity', or 'dpsgd'")
    parser.add_argument("--optimizer", default="lbfgs", type=str,
                        help="optimizer used for training: 'lbfgs' (default) or 'sgd'")
    parser.add_argument("--num_models", default=32, type=int,
                        help="number of models to train (for subsagg; default = 32)")
    parser.add_argument("--num_repetitions", default=10, type=int,
                        help="number of times to repeat experiment (default = 10)")
    parser.add_argument("--epsilon", default=math.inf, type=float,
                        help="privacy loss for predictions (default = infinity)")
    parser.add_argument("--delta", default=0.0, type=float,
                        help="privacy failure probability for predictions (default = 0.0)")
    parser.add_argument("--inference_budget", default=-1, type=int,
                        help="number of inferences to support (default = -1 for all)")
    parser.add_argument("--num_epochs", default=100, type=int,
                        help="number of training epochs (default = 100)")
    parser.add_argument("--batch_size", default=32, type=int,
                        help="batch size for SGD (default = 32)")
    parser.add_argument("--noise_dist", default="sqrt_gaussian", type=str,
                        help="noise distribution for sensitivity methods "
                        "methods: 'sqrt_gaussian' (default), 'laplacian', "
                        "'gaussian', 'advanced_gaussian'")
    parser.add_argument("--learning_rate", default=1., type=float,
                        help="initial learning rate for SGD (default = 1.0)")
    parser.add_argument("--weight_decay", default=0., type=float,
                        help="L2 regularization parameter (default = "
                        "0.0; set to -1 to cross-validate)")
    parser.add_argument("--clip", default=1e-1, type=float,
                        help="gradient clipping for DP-SGD method (default "
                        "= 1e-1; set to -1 to cross-validate)")
    parser.add_argument("--num_samples", default=-1, type=int,
                        help="number of training samples (default: all)")
    parser.add_argument("--num_classes", default=-1, type=int,
                        help="number of classes to use (default: all)")
    parser.add_argument("--pca_dims", default=-1, type=int,
                        help="number of PCA dimensions for data (default: not used)")
    parser.add_argument("--data_folder", default="/tmp", type=str,
                        help="folder in which to store data (default: '/tmp')")
    parser.add_argument("--result_file", default=None, type=str,
                        help="file in which to write experimental results")
    parser.add_argument("--device", default="cpu", type=str,
                        help="compute device: 'cpu' (default) or 'gpu'")
    parser.add_argument("--use_lr_scheduler", default=False, action="store_true",
                        help="use learning rate reduction (for DP-SGD)")
    parser.add_argument("--visdom", default=hostname, type=str,
                        help=f"visdom server (default = {hostname})")
    args = parser.parse_args()

    # check generic input arguments:
    if args.epsilon < 0.:
        raise ValueError("Epsilon value must be non-negative.")

    # run:
    main(args)
