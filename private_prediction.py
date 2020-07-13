#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import torch
import torch.nn as nn

import dpsgd_privacy
import modeling
import noise
import util

INFERENCE_BUDGETS = [1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 250, 300,
                     400, 500, 750, 1000]


def split_dataset(dataset, num_splits):
    """
    Splits a dataset into `num_splits` disjoint subsets.
    """
    assert num_splits >= 1, "number of splits must be positive integer"

    # assign examples to splits:
    N = dataset["features"].size(0)
    split_size = N // num_splits
    indices = torch.randperm(N)[:num_splits * split_size]
    indices = indices.view(split_size, num_splits)
    indices = indices.to(dataset["features"].device)

    # create datasets according to splits:
    datasets = [{key: dataset[key].index_select(0, indices[:, idx])
                 for key in dataset.keys()} for idx in range(num_splits)]
    return datasets


def private_prediction(predictions, epsilon=0.):
    """
    Performs private prediction for N examples given an NxK matrix `predictions`
    that contains K predictions for each example, which were produced by K
    classifiers trained on disjoint training sets.

    The parameter `epsilon` controls the privacy of the prediction: a value of
    0 (default) implies maximum privacy by randomly picking a class, and a value
    of `math.inf` performs a majority vote over the K predictions.

    The private prediction algorithm used is described in Dwork & Feldman (2018).
    """
    assert predictions.dim() == 2, "predictions must be 2D matrix"
    assert epsilon >= 0., "epsilon cannot be negative"

    # count the votes in the predictions:
    N, K = predictions.size()
    num_classes = predictions.max() + 1
    counts = torch.zeros(N, num_classes)
    for c in range(num_classes):
        counts[:, c] = (predictions == c).sum(dim=1)

    # perform private prediction by sampling from smoothed Gibbs distribution on counts:
    if epsilon == math.inf:
        return counts.argmax(dim=1)
    else:
        logits = counts.mul(epsilon)
        probabilities = logits.sub(logits.max(dim=1, keepdim=True).values).exp_()
        probabilities.div_(probabilities.sum(dim=1, keepdim=True))
        return noise.categorical(probabilities)


def get_b_function(epsilon, delta, supremum=True):
    """
    Helper function that returns the B function used in the advanced Gaussian
    mechanism of Balle & Wang (2018).
    """
    gaussian = torch.distributions.normal.Normal(0, 1)

    def b_function(v):
        term = math.exp(epsilon) * gaussian.cdf(-math.sqrt(epsilon * (v + 2)))
        if supremum:
            return gaussian.cdf(math.sqrt(epsilon * v)) - term
        else:
            return -gaussian.cdf(-math.sqrt(epsilon * v)) + term

    return b_function


def sensitivity_scale(epsilon, delta, weight_decay,
                      criterion, dataset_size, noise_dist,
                      chaudhuri=True):
    """
    Given differential privacy parameters `epsilon` and `delta`, L2
    regularization parameter `weight_decay`, the specified `criterion`, dataset
    size `dataset_size`, and the noise distribution `noise_dist` compute the
    `scale` of the distribution to be used for the model and logit sensitivity
    methods.

    If `chaudhuri` is True, we use assumptions from Chaudhuri et al. to compute
    the scale.
    """
    if noise_dist in ["gaussian", "advanced_gaussian"]:
        if delta <= 0:
            raise ValueError(f"Delta must be > 0 for Gaussian noise (not {delta}).")
    elif delta != 0:
        raise ValueError(f"Delta must be 0 for non-Gaussian noise (not {delta}).")

    # standard Gaussian mechanism of Dwork (2014):
    if noise_dist == "gaussian":
        if epsilon < 0 or epsilon > 1:
            raise ValueError(
                f"Epsilon must be in (0, 1) for Gaussian noise (not {epsilon}).")
        scale = epsilon / math.sqrt(2 * math.log(1.25 / delta))

    # advanced Gaussian mechanism of Balle and Wang (2018):
    elif noise_dist == "advanced_gaussian":

        # compute delta knot:
        gaussian = torch.distributions.normal.Normal(0, 1)
        delta0 = gaussian.cdf(0) - math.exp(epsilon) * gaussian.cdf(-math.sqrt(2. * epsilon))

        # define B-function:
        supremum = (delta >= delta0)
        b_func = get_b_function(epsilon, delta, supremum=supremum)

        # define constraint on output of B-function:
        def constraint(x):
            return x <= delta if supremum else x < -delta

        # find maximum value of B-function:
        try:
            maximum = next(2 ** k for k in range(128) if not constraint(b_func(2 ** k)))
        except StopIteration:
            logging.error("Optimal value for v* out of range [0, 2 ** 128].")
        tol = 1e-5
        v_star = util.binary_search(b_func, constraint, 0, maximum, tol=tol)

        # compute noise multiplier:
        if supremum:
            alpha = math.sqrt(1 + v_star / 2.0) - math.sqrt(v_star / 2.0)
        else:
            v_star += tol  # binary search returns value that is slightly too small
            alpha = math.sqrt(1 + v_star / 2.0) + math.sqrt(v_star / 2.0)
        scale = math.sqrt(2. * epsilon) / alpha

    # standard bounds for exponential / gamma mechanism:
    elif noise_dist == "laplacian" or noise_dist == "sqrt_gaussian":
        scale = epsilon
    else:
        raise ValueError(f"Unknown noise distribution: {noise_dist}")

    # computes the Lipschitz constant for a given loss:
    if isinstance(criterion, nn.CrossEntropyLoss):
        k = math.sqrt(2.0)
    elif isinstance(criterion, nn.BCELoss):
        k = 1.0
    else:
        raise ValueError("Lipschitz constant of loss unknown.")

    # compute final sensitivity scale:
    if chaudhuri:
        scale *= (weight_decay * dataset_size / (2.0 * k))
    return scale


def advanced_compose(epsilon, delta, budget, del_prime):
    """
    Applies the advanced composition of Theorem 1.1 of "Concentrated
    Differential Privacy", Dwork and Rothblum, 2016.

    Computes the epsilon and delta for a single application of the
    differentially private mechanism such that after `budget` compositions, the
    composed mechanism satisfies (`epsilon`, `delta`)-differential privacy. The
    argument `del_prime` parameterizes the trade-off between the computed
    epsilon and delta and is valid in the range `(0, delta]`.
    """
    assert del_prime > 0, "del_prime must be > 0."
    assert del_prime <= delta, "del_prime must be <= global delta."

    log_dp = math.log(1 / del_prime)
    eps_ind = math.sqrt(log_dp + epsilon) - math.sqrt(log_dp)
    eps_ind *= math.sqrt(2 / budget)
    del_ind = (delta - del_prime) / budget
    return eps_ind, del_ind


def loss_perturbation_params(
    epsilon, delta, noise_dist, criterion, dataset_size, num_classes
):
    """
    Given differential privacy parameters `epsilon` and `delta`, the specified
    noise distribution `noise_dist`, the specified `criterion`, dataset size
    `dataset_size` and number of classes `num_classes`, compute the `precision`
    of the distribution and the `weight_decay` to be used for loss
    perturbation.
    """
    assert epsilon > 0., "epsilon must be positive"

    # lamb_max is a bound on the eigenvalues of the hessian of loss
    # C is a bound on the rank of the hessian of the loss (which is typically the
    # number of classes but may be less).
    # K is a bound on the lipschitz constant of the loss function
    if isinstance(criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
        k = math.sqrt(2)
        C = num_classes
        lamb_max = 0.5
    elif isinstance(criterion, (nn.BCELoss, nn.BCEWithLogitsLoss)):
        k = 1.0
        C = 1.0
        lamb_max = 0.25
    else:
        raise ValueError("Required constants for loss function are unknown.")

    if noise_dist == "sqrt_gaussian":
        if delta != 0:
            raise ValueError(
                f"Delta must be zero for sqrt_gaussian noise, not {delta}.")
        noise_mul = 0.5
    elif noise_dist == "gaussian":
        if delta <= 0:
            raise ValueError(
                f"Delta cannot be zero for gaussian noise, not {delta}.")
        noise_mul = 1.0 / math.sqrt(8 * math.log(2 / delta) + 4 * epsilon)
    else:
        raise ValueError("Invalid noise distribution")

    precision = noise_mul * epsilon / k
    weight_decay = 2 * lamb_max * C / (epsilon * dataset_size)
    return precision, weight_decay


def subsagg_method(data, args, visualizer=None, title=None):
    """
    Given a dataset `data` and arguments `args`, run a full test of the private
    prediction algorithm of Dwork & Feldman (2018). Returns a `dict` containing
    the `predictions` for the training and test data.
    """

    # unspecified inference budgets means we are trying many values:
    if args.inference_budget == -1:
        inference_budgets = INFERENCE_BUDGETS
    else:
        inference_budgets = [args.inference_budget]

    # split training set into disjoint subsets:
    data["split_train"] = split_dataset(data["train"], args.num_models)

    # train all classifiers:
    logging.info(f"Training {args.num_models} disjoint classifiers...")
    models = [None] * args.num_models
    for idx in range(args.num_models):

        # initialize model:
        logging.info(f" => training model {idx + 1} of {args.num_models}:")
        num_classes = int(data["train"]["targets"].max()) + 1
        num_features = data["split_train"][idx]["features"].size(1)
        models[idx] = modeling.initialize_model(
            num_features, num_classes, model=args.model, device=args.device
        )

        # train using L2-regularized loss:
        regularized_criterion = modeling.add_l2_regularization(
            nn.CrossEntropyLoss(), models[idx], args.weight_decay
        )
        augmentation = (args.model != "linear")
        modeling.train_model(models[idx], data["split_train"][idx],
                             criterion=regularized_criterion,
                             optimizer=args.optimizer,
                             num_epochs=args.num_epochs,
                             learning_rate=args.learning_rate,
                             batch_size=args.batch_size,
                             augmentation=augmentation,
                             visualizer=visualizer,
                             title=title)

    # clean up:
    del data["split_train"]

    # perform inference on both training and test set:
    logging.info("Performing inference with private predictor...")
    predictions = {}
    for split in data.keys():

        # compute predictions of each model:
        batch_size = data[split]["targets"].size(0) if args.model == "linear" else 128
        preds = [modeling.test_model(
            model, data[split], augmentation=augmentation, batch_size=batch_size,
        ) for model in models]
        preds = [pred.argmax(dim=1) for pred in preds]
        preds = torch.stack(preds, dim=1)

        # compute private predictions:
        if split not in predictions:
            predictions[split] = {}
        for inference_budget in inference_budgets:
            # privacy parameter must be corrected for inference budget:
            epsilon = args.epsilon / float(inference_budget)
            if args.delta > 0:
                eps, _ = advanced_compose(
                    args.epsilon, args.delta, inference_budget, args.delta)
                epsilon = max(eps, epsilon)

            # compute and store private predictions:
            predictions[split][inference_budget] = \
                private_prediction(preds, epsilon=epsilon)

    # return predictions:
    return predictions


def loss_perturbation_method(data, args, visualizer=None, title=None):
    """
    Given a dataset `data` and arguments `args`, run a full test of the private
    prediction algorithms of Chaudhuri et al. (2011) / Kifer et al. (2012)
    generalized to the multi-class setting. Returns a `dict` containing the
    `predictions` for the training and test data.

    Note: This algorithm only guarantees privacy under the following assumptions:
    - The loss is strictly convex and has a continuous Hessian.
    - The model is linear.
    - The inputs have a 2-norm restricted to be less than or equal 1.
    - The Lipschitz constant of the loss function and the spectral
        norm of the Hessian must be bounded.
    """
    assert args.model == "linear", f"Model {args.model} not supported."
    assert args.noise_dist != "advanced_gaussian", \
        "Advanced Gaussian method not supported for loss perturbation."

    # get dataset properties:
    num_classes = int(data["train"]["targets"].max()) + 1
    num_samples, num_features = data["train"]["features"].size()

    # initialize model and criterion:
    model = modeling.initialize_model(num_features, num_classes, device=args.device)
    criterion = nn.CrossEntropyLoss()

    precision, weight_decay = loss_perturbation_params(
        args.epsilon, args.delta, args.noise_dist,
        criterion, num_samples, num_classes)
    weight_decay = max(weight_decay, args.weight_decay)

    # sample loss perturbation vector:
    param = modeling.get_parameter_vector(model)
    mean = torch.zeros_like(param)
    perturbation = getattr(noise, args.noise_dist)(mean, precision)
    perturbations = [torch.zeros_like(p) for p in model.parameters()]
    modeling.set_parameter_vector(perturbations, perturbation)

    # closure implementing the loss-perturbation criterion:
    def loss_perturbation_criterion(predictions, targets):
        loss = criterion(predictions, targets)
        for param, perturb in zip(model.parameters(), perturbations):
            loss += ((param * perturb).sum() / num_samples)
        return loss

    # add L2-regularizer to the loss:
    regularized_criterion = modeling.add_l2_regularization(
        loss_perturbation_criterion, model, weight_decay
    )

    # train classifier:
    logging.info("Training classifier with loss perturbation...")
    modeling.train_model(model, data["train"],
                         criterion=regularized_criterion,
                         optimizer=args.optimizer,
                         num_epochs=args.num_epochs,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         visualizer=visualizer,
                         title=title)

    # perform inference on both training and test set:
    logging.info("Performing inference with loss-perturbed predictor...")
    predictions = {split: model(data_split["features"]).argmax(dim=1)
                   for split, data_split in data.items()}
    return predictions


def model_sensitivity_method(data, args, visualizer=None, title=None):
    """
    Given a dataset `data` and arguments `args`, run a full test of private
    prediction using the model sensitivity method.

    Note: This algorithm only guarantees privacy for models with convex losses.
    """
    assert args.model == "linear", f"Model {args.model} not supported."

    # initialize model and criterion:
    num_classes = int(data["train"]["targets"].max()) + 1
    num_samples, num_features = data["train"]["features"].size()
    model = modeling.initialize_model(num_features, num_classes, device=args.device)
    criterion = nn.CrossEntropyLoss()
    regularized_criterion = modeling.add_l2_regularization(
        criterion, model, args.weight_decay
    )

    # train classifier:
    logging.info("Training non-private classifier...")
    modeling.train_model(model, data["train"],
                         criterion=regularized_criterion,
                         optimizer=args.optimizer,
                         num_epochs=args.num_epochs,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         visualizer=visualizer,
                         title=title)

    # perturb model parameters:
    logging.info("Applying model sensitivity method...")
    scale = sensitivity_scale(args.epsilon, args.delta, args.weight_decay,
                              criterion, num_samples, args.noise_dist)
    param = modeling.get_parameter_vector(model)
    mean = torch.zeros_like(param)
    noise_dist = "gaussian" if args.noise_dist in ["gaussian", "advanced_gaussian"] \
        else args.noise_dist
    perturbation = getattr(noise, noise_dist)(mean, scale)

    with torch.no_grad():
        param.add_(perturbation)
    modeling.set_parameter_vector(model, param)

    # perform inference on both training and test set:
    logging.info("Performing inference with perturbed predictor...")
    predictions = {split: modeling.test_model(model, data_split).argmax(dim=1)
                   for split, data_split in data.items()}
    return predictions


def logit_sensitivity_method(data, args, visualizer=None, title=None):
    """
    Given a dataset `data` and arguments `args`, run a full test of the logit
    sensitivity method. Returns a `dict` containing the `predictions` for the
    training and test data.

    Note: This algorithm only guarantees privacy for models with convex losses.
    """
    assert args.model == "linear", f"Model {args.model} not supported."

    # unspecified inference budgets means we are trying many values:
    if args.inference_budget == -1:
        inference_budgets = INFERENCE_BUDGETS
    else:
        inference_budgets = [args.inference_budget]

    # initialize model and criterion:
    num_classes = int(data["train"]["targets"].max()) + 1
    num_samples, num_features = data["train"]["features"].size()
    model = modeling.initialize_model(num_features, num_classes, device=args.device)
    criterion = nn.CrossEntropyLoss()
    regularized_criterion = modeling.add_l2_regularization(
        criterion, model, args.weight_decay
    )

    # train classifier:
    logging.info("Training non-private classifier...")
    modeling.train_model(model, data["train"],
                         criterion=regularized_criterion,
                         optimizer=args.optimizer,
                         num_epochs=args.num_epochs,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         visualizer=visualizer,
                         title=title)

    # perform inference on both training and test set:
    logging.info("Performing inference with private predictor...")
    predictions = {}
    for split in data.keys():
        if split not in predictions:
            predictions[split] = {}
        for inference_budget in inference_budgets:

            # account for the budget in the noise scale:
            scale = sensitivity_scale(
                args.epsilon / float(inference_budget),
                args.delta / float(inference_budget), args.weight_decay,
                criterion, num_samples, args.noise_dist)
            if args.delta > 0:
                # linearly search for the optimal noise scale under advanced
                # composition:
                del_primes = torch.linspace(0, args.delta, 1000)[1:-1]
                ind_eps_del = [advanced_compose(
                    args.epsilon, args.delta, inference_budget, dp)
                    for dp in del_primes]
                scales = [sensitivity_scale(
                    epsilon, delta, args.weight_decay,
                    criterion, num_samples, args.noise_dist)
                    for epsilon, delta in ind_eps_del]
                # for small budgets the naive scale may be better:
                scale = max(max(scales), scale)

            # make private predictions:
            noise_dist = "gaussian" if args.noise_dist in ["gaussian", "advanced_gaussian"] \
                else args.noise_dist
            preds = modeling.test_model(model, data[split])
            mean = torch.zeros_like(preds).T
            preds += getattr(noise, noise_dist)(mean, scale).T

            # make private predictions:
            predictions[split][inference_budget] = preds.argmax(dim=1)

    # return predictions:
    return predictions


def dpsgd_method(data, args, visualizer=None, title=None):
    """
    Given a dataset `data` and arguments `args`, run a full test of private
    prediction using the differentially private SGD training method of dpsgd
    et al. (2016).
    """

    # assertions:
    if args.optimizer != "sgd":
        raise ValueError(f"DP-SGD does not work with {args.optimizer} optimizer.")
    if args.delta <= 0.:
        raise ValueError(f"Specified delta must be positive (not {args.delta}).")

    # initialize model and criterion:
    num_classes = int(data["train"]["targets"].max()) + 1
    num_samples = data["train"]["features"].size(0)
    num_features = data["train"]["features"].size(1)
    model = modeling.initialize_model(
        num_features, num_classes, model=args.model, device=args.device
    )
    regularized_criterion = modeling.add_l2_regularization(
        nn.CrossEntropyLoss(), model, args.weight_decay
    )

    # compute standard deviation of noise to add to gradient:
    num_samples = data["train"]["features"].size(0)
    std, eps = dpsgd_privacy.compute_noise_multiplier(
        args.epsilon, args.delta, num_samples, args.batch_size, args.num_epochs)
    logging.info(f"DP-SGD with noise multiplier (sigma) of {std}.")
    logging.info(f"Epsilon error is {abs(eps - args.epsilon):.5f}.")

    # convert model to make differentially private gradient updates:
    model = modeling.privatize_model(model, args.clip, std)

    # train classifier:
    logging.info("Training classifier using private SGD...")
    augmentation = (args.model != "linear")
    modeling.train_model(model, data["train"],
                         optimizer=args.optimizer,
                         criterion=regularized_criterion,
                         num_epochs=args.num_epochs,
                         learning_rate=args.learning_rate,
                         batch_size=args.batch_size,
                         momentum=0.0,
                         use_lr_scheduler=args.use_lr_scheduler,
                         augmentation=augmentation,
                         visualizer=visualizer,
                         title=title)

    # convert model back to "regular" model:
    model = modeling.unprivatize_model(model)

    # perform inference on both training and test set:
    logging.info("Performing inference with DP-SGD predictor...")
    predictions = {split: modeling.test_model(
                   model, data_split, augmentation=augmentation
                   ).argmax(dim=1) for split, data_split in data.items()}
    return predictions


def compute_accuracy(args, data, accuracies=None, visualizer=None):
    """
    Runs a single experiment using the settings in `args` on the specified
    `data`. Accuracies resulting from the experiment are stored in `accuracies`.

    If a visdom `visualizer` is specified, the function plots learning curves.
    """

    # check inputs:
    if accuracies is None:
        accuracies = {}
    else:
        assert isinstance(accuracies, dict), "accuracies must be dict"

    # run the specified private prediction method:
    title = "Learning curve"
    method_name = f"{args.method}_method"
    if method_name not in globals():
        raise ValueError(f"Unknown private prediction method: {args.method}")
    method_func = globals()[method_name]
    predictions = method_func(data, args, visualizer=visualizer, title=title)

    # compute accuracy on all splits:
    for split, preds in predictions.items():

        # get targets for this split:
        targets = data[split]["targets"]

        # prediction accuracy independent of inference budget:
        if torch.is_tensor(preds):

            # make sure predictions and targets live on the same device:
            if preds.device != targets.device:
                preds = preds.to(device=targets.device)

            # compute accuracy:
            if split not in accuracies:
                accuracies[split] = []
            accuracy = float(preds.eq(targets).sum()) / targets.size(0)
            logging.info(f" => {split} accuracy: {accuracy}")
            accuracies[split].append(accuracy)

        # prediction accuracy depends on inference budget:
        elif isinstance(preds, dict):
            if split not in accuracies:
                accuracies[split] = {}
            for budget, budget_preds in preds.items():

                # make sure predictions and targets live on the same device:
                if budget_preds.device != targets.device:
                    budget_preds = budget_preds.to(device=targets.device)

                # compute accuracy:
                budget = str(budget)
                if budget not in accuracies[split]:
                    accuracies[split][budget] = []
                accuracy = float(budget_preds.eq(targets).sum()) / targets.size(0)
                logging.info(f" => {split} accuracy at {budget} budget: {accuracy}")
                accuracies[split][budget].append(accuracy)

        # this should never happen:
        else:
            raise ValueError("Unknown format of preds variable.")

    # return:
    return accuracies
