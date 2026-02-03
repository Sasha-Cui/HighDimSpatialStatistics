"""Marginal parameter fitting routines."""
from __future__ import annotations

from typing import List, Tuple

import torch

from HighDimSpatial.kernels.approx import approx_matern_kernel_marginal
from HighDimSpatial.metrics.likelihood import negative_log_likelihood


def _split_groups(X: torch.Tensor, Y: torch.Tensor, number_of_groups: int):
    n_locations = X.size(0)
    group_size = max(1, n_locations // number_of_groups)
    X_groups = torch.split(X, group_size)
    Y_groups = torch.split(Y, group_size)
    return X_groups, Y_groups


def optimize_marginal_parameters(
    X: torch.Tensor,
    Y: torch.Tensor,
    number_of_groups: int,
    number_of_cycles: int = 20,
    steps_per_batch: int = 1,
    print_early_stopping_epochs: bool = False,
    plot_loss: bool = False,
) -> list[tuple[float, float, float]]:
    """Optimize alpha/nu/sigma for each variable using batch learning."""
    p = Y.size(1)
    X_groups, Y_groups = _split_groups(X, Y, number_of_groups)

    optimized_params: list[tuple[float, float, float]] = []

    for i in range(p):
        loss_history = []
        alpha_i = torch.tensor(0.1, dtype=torch.float64, device=X.device).requires_grad_(True)
        nu_i = torch.tensor(0.9, dtype=torch.float64, device=X.device).requires_grad_(True)
        sigma_i = torch.tensor(1.0, dtype=torch.float64, device=X.device).requires_grad_(True)

        tolerance = 1e-15
        patience = 25
        best_loss = float("inf")
        epochs_no_improve = 0
        times_lr_reduced = 0

        optimizer = torch.optim.Adagrad(
            [
                {"params": alpha_i, "lr": 0.0003},
                {"params": nu_i, "lr": 0.008},
                {"params": sigma_i, "lr": 0.25},
            ],
            lr_decay=0,
            weight_decay=0,
            eps=1e-15,
        )

        for epoch in range(number_of_cycles):
            total_nll = 0.0
            try:
                for X_batch, Y_batch in zip(X_groups, Y_groups):
                    for _ in range(steps_per_batch):
                        K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                        nll = negative_log_likelihood(Y_batch[:, i], K)
                        total_nll += nll.item()

                        optimizer.zero_grad()
                        nll.backward()
                        optimizer.step()

                        with torch.no_grad():
                            alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                            nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                            sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)

                loss_history.append(total_nll)
                if total_nll < best_loss - tolerance:
                    best_loss = total_nll
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1

                if epochs_no_improve >= patience:
                    for param_group in optimizer.param_groups:
                        param_group["lr"] *= 0.1
                        times_lr_reduced += 1
                    if times_lr_reduced >= 3:
                        break
                    if print_early_stopping_epochs:
                        print("Reducing learning rate by 0.1 at epoch", epoch)
                    epochs_no_improve = 0

            except Exception as exc:
                print(f"Error during epoch {epoch}: {exc}")
                print(f"Parameters: alpha={alpha_i}, nu={nu_i}, sigma={sigma_i}")
                break

        if plot_loss:
            import matplotlib.pyplot as plt

            plt.plot(loss_history)
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Loss over Epochs")
            plt.show()

        optimized_params.append((alpha_i.item(), nu_i.item(), sigma_i.item()))

    return optimized_params


def optimize_marginal_parameters_in_groups(
    lr_set: dict,
    init_set: dict,
    X_groups: list[torch.Tensor],
    Y_groups: list[torch.Tensor],
    number_of_cycles: int = 1,
    steps_per_batch: int = 1,
    print_early_stopping_epochs: bool = False,
    sigma_is_known: bool = True,
) -> tuple[list[list[torch.Tensor]], list[dict], list[list[float]]]:
    """Optimize marginal parameters using provided groups."""
    p = Y_groups[0].size(1)
    optimized_params = []
    best_params = []
    loss_histories = []

    for i in range(p):
        loss_history = []

        alpha_i = torch.tensor(init_set["alpha_init"], dtype=torch.float64, device=Y_groups[0].device).requires_grad_(True)
        nu_i = torch.tensor(init_set["nu_init"], dtype=torch.float64, device=Y_groups[0].device).requires_grad_(True)

        if sigma_is_known:
            sigma_i = torch.tensor(1.0, dtype=torch.float64, device=Y_groups[0].device).requires_grad_(False)
        else:
            sigma_i = torch.tensor(init_set["sigma_init"], dtype=torch.float64, device=Y_groups[0].device).requires_grad_(True)

        tolerance = 1e-15
        patience = 10
        best_loss = float("inf")
        epochs_no_improve = 0

        params = [alpha_i, nu_i] + ([] if sigma_is_known else [sigma_i])
        optimizer = torch.optim.Adagrad(
            [
                {"params": alpha_i, "lr": lr_set["alpha_lr"]},
                {"params": nu_i, "lr": lr_set["nu_lr"]},
                *([] if sigma_is_known else [{"params": sigma_i, "lr": lr_set["sigma_lr"]}]),
            ],
            lr_decay=0,
            weight_decay=0,
            eps=1e-15,
        )

        best_param = {"alpha": alpha_i.clone(), "nu": nu_i.clone(), "sigma": sigma_i.clone()}

        for epoch in range(number_of_cycles):
            total_nll = 0.0
            for X_batch, Y_batch in zip(X_groups, Y_groups):
                for _ in range(steps_per_batch):
                    K = approx_matern_kernel_marginal(X_batch, alpha_i, nu_i, sigma_i)
                    nll = negative_log_likelihood(Y_batch[:, i], K)
                    total_nll += nll.item()

                    optimizer.zero_grad()
                    nll.backward()
                    optimizer.step()

                    with torch.no_grad():
                        alpha_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                        nu_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)
                        if not sigma_is_known:
                            sigma_i.clamp_(min=torch.finfo(torch.float64).eps, max=50)

            loss_history.append(total_nll)
            if total_nll < best_loss - tolerance:
                best_loss = total_nll
                epochs_no_improve = 0
                best_param = {"alpha": alpha_i.clone(), "nu": nu_i.clone(), "sigma": sigma_i.clone()}
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                if print_early_stopping_epochs:
                    print("Reducing learning rate at epoch", epoch)
                for param_group in optimizer.param_groups:
                    param_group["lr"] *= 0.1
                epochs_no_improve = 0

        loss_histories.append(loss_history)
        best_params.append(best_param)
        optimized_params.append([alpha_i, nu_i, sigma_i])

    return optimized_params, best_params, loss_histories
