import signal
from collections import defaultdict

import numpy as np
import torch
from lakde.callbacks import ELBOCallback, LikelihoodCallback
from lakde.kdes import AbstractKDE
from lakde.local_adaptive_kde import LocallyAdaptiveKDE


def run_kde_model(
    model,
    train_data,
    val_data,
    test_data,
    *,
    threshold,
    ll_rtol=1e-4,
    max_iterations=1000,
    validate_every=1,
):
    train_data = torch.from_numpy(train_data).float().cuda()
    val_data = torch.from_numpy(val_data).float().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()

    N = train_data.shape[0]
    max_active_rnms = 0

    likelihood_callback = LikelihoodCallback(val_data, rtol=ll_rtol, verbose=True)

    def maximum_active_responsibilities(X, model, iter_step):
        nonlocal max_active_rnms
        if hasattr(model, "rnm_active_contribs"):
            active_rnms = model.rnm_active_contribs.sum().float() / N**2
        else:
            active_rnms = 0
        if active_rnms > max_active_rnms:
            max_active_rnms = active_rnms

    callbacks = [likelihood_callback, maximum_active_responsibilities]
    if isinstance(model, LocallyAdaptiveKDE):
        callbacks.append(ELBOCallback())
    model.fit(train_data, max_iterations, callbacks, threshold)

    test_ll = model.log_pred_density(train_data, test_data).mean().item()
    return test_ll, {"max_active": max_active_rnms}


def run_k_radius_model(model, train_data, test_data):
    from experiments.utils import density_mesh, trapz_grid

    train_data = torch.from_numpy(train_data).float().cuda()
    test_data = torch.from_numpy(test_data).float().cuda()

    assert train_data.shape[-1] == 2
    corrected = model.corrected

    try:
        # numerically find the normalization constant, only works in 2 dimensions
        model.corrected = True
        corrected_const = trapz_grid(*density_mesh(model, train_data)).cpu().item()
        corrected_ll = model.log_pred_density(train_data, test_data).mean().cpu().item()

        model.corrected = False
        uncorrected_const = trapz_grid(*density_mesh(model, train_data)).cpu().item()
        uncorrected_ll = (
            model.log_pred_density(train_data, test_data).mean().cpu().item()
        )

    finally:
        model.corrected = corrected
    return corrected_ll - np.log(corrected_const), {
        "corrected_ll": corrected_ll,
        "corrected_const": corrected_const,
        "uncorrected_ll": uncorrected_ll,
        "uncorrected_const": uncorrected_const,
    }


def get_subset(dataset, size, seed):
    rng = np.random.default_rng(seed)
    sample_size = min(len(dataset), size)
    inds = rng.choice(len(dataset), sample_size, replace=False)
    return dataset[inds]


def run_model(
    model,
    train_data,
    val_data,
    test_data,
    *,
    threshold,
    ll_rtol=1e-4,
    max_iterations=1000,
    validate_every=1,
):
    if isinstance(model, AbstractKDE):
        return run_kde_model(
            model,
            train_data,
            val_data,
            test_data,
            threshold=threshold,
            ll_rtol=ll_rtol,
            max_iterations=max_iterations,
            validate_every=validate_every,
        )
    else:
        return run_k_radius_model(model, train_data, test_data)


def run_all(
    train_data_supplier,
    val_data,
    test_data,
    model_supplier,
    sample_sizes,
    num_subsets,
    **kwargs,
):
    metrics = defaultdict(list)

    def noop_handler(sig, frame):
        pass

    old_handler = signal.signal(signal.SIGINT, noop_handler)

    try:
        for sample_size in sample_sizes:
            for k in range(num_subsets):
                train_data = train_data_supplier(sample_size)
                model = model_supplier(sample_size)

                print(
                    "Training model on {} sample subset ({}/{})".format(
                        sample_size, k + 1, num_subsets
                    )
                )
                log_density, meta = run_model(
                    model, train_data, val_data, test_data, **kwargs
                )
                print("Finished with log density {}".format(log_density))

                metrics[str(sample_size)].append(log_density)

                if meta is not None:
                    for k, v in meta.items():
                        metrics[str(sample_size) + "_" + k].append(v)

        return metrics
    finally:
        signal.signal(signal.SIGINT, old_handler)
