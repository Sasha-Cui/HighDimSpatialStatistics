"""Cross-parameter fitting routines."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from HighDimSpatial.config import resolve_data_path
from HighDimSpatial.kernels.approx import approx_matern_kernel_cross, approx_matern_kernel_cross_legacy
from HighDimSpatial.kernels.matern import compute_parameter_matrices
from HighDimSpatial.metrics.likelihood import negative_log_likelihood
from HighDimSpatial.utils.linalg import is_positive_definite


def save_checkpoint_cross(
    epoch: int,
    Delta_A: torch.Tensor,
    Delta_B: torch.Tensor,
    rho_A: torch.Tensor,
    rho_B: torch.Tensor,
    rho_V: torch.Tensor,
    W: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    best_loss: float,
    filename: str,
) -> None:
    checkpoint_data = {
        "epoch": epoch,
        "Delta_A": Delta_A.clone().detach(),
        "Delta_B": Delta_B.clone().detach(),
        "rho_A": rho_A.clone().detach(),
        "rho_B": rho_B.clone().detach(),
        "rho_V": rho_V.clone().detach(),
        "W": W.clone().detach(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(checkpoint_data, filename)


def load_checkpoint_cross(filename: str):
    if os.path.exists(filename):
        return torch.load(filename, weights_only=True)
    return None


def _default_checkpoint_path() -> Path:
    path = resolve_data_path("processed", "checkpoints", "cross_checkpoint.pth")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def optimize_cross_parameters_in_groups(
    optimized_marginal_params: list,
    lr_set: dict,
    X_groups: list[torch.Tensor],
    Y_groups: list[torch.Tensor],
    number_of_cycles: int = 300,
    steps_per_batch: int = 2,
    print_early_stopping_epochs: bool = False,
    checkpoint_interval: int = 50,
    max_time_hours: float = 24,
    logging: bool = False,
    checkpoint_path: str | None = None,
    use_legacy_kernel: bool = False,
):
    """Optimize cross-covariance parameters using batch learning with group-wise updates."""

    def perform_optimization_step_with_halving(
        optimizer,
        params,
        param_names,
        model_state_before_step,
        grad_state_before_step,
        max_halving_attempts,
        epoch,
        step,
        halving_log,
        lr_log,
        Delta_A,
        Delta_B,
        rho_A,
        rho_B,
        rho_V,
        W,
        alpha,
        nu,
        sigma,
        X_batch,
    ):
        success = False
        halving_attempts = 0
        while not success and halving_attempts < max_halving_attempts:
            optimizer.step()

            with torch.no_grad():
                eps = torch.finfo(torch.float64).eps
                params_min_max = {
                    rho_A: (eps, 1 - eps),
                    rho_B: (eps, 1 - eps),
                    rho_V: (-1 + eps, 1 - eps),
                    W: (eps, 1 - eps),
                    Delta_A: (eps, None),
                    Delta_B: (eps, None),
                }
                for param, (min_val, max_val) in params_min_max.items():
                    param.clamp_(min=min_val, max=max_val)

            alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(
                Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma
            )
            K = (
                approx_matern_kernel_cross_legacy(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                if use_legacy_kernel
                else approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
            )

            if is_positive_definite(K):
                success = True
            else:
                print(f"Condition failed at epoch {epoch}, halving learning rate. Attempt {halving_attempts + 1}")
                halving_log.append(f"Epoch {epoch}: LR halved at attempt {halving_attempts + 1}")

                with torch.no_grad():
                    for name, param in zip(param_names, params):
                        param.copy_(model_state_before_step[name])
                        param.grad.copy_(grad_state_before_step[f"{name}_grad"])

                for param_group in optimizer.param_groups:
                    param_group["lr"] /= 2
                halving_attempts += 1

        lr_log.append(optimizer.param_groups[0]["lr"])
        return success

    start_time = time.time()
    max_time_seconds = max_time_hours * 3600

    tolerance = 1e-15
    patience = 5
    epochs_no_improve = 0
    best_loss = float("inf")

    p = Y_groups[0].size(1)
    device = Y_groups[0].device

    Delta_A = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    Delta_B = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_A = torch.tensor(1 - torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_B = torch.tensor(1 - torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    rho_V = torch.tensor(-torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)
    W = torch.full((p,), torch.finfo(torch.float64).eps, dtype=torch.float64, device=device, requires_grad=True)

    alpha = torch.tensor([param[0] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    nu = torch.tensor([param[1] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)
    sigma = torch.tensor([param[2] for param in optimized_marginal_params], dtype=torch.float64, requires_grad=False, device=device)

    param_names = ["Delta_A", "Delta_B", "rho_A", "rho_B", "rho_V", "W"]
    params = [Delta_A, Delta_B, rho_A, rho_B, rho_V, W]

    optimizer = torch.optim.Adam([
        {"params": param, "lr": lr_set[name]} for name, param in zip(param_names, params)
    ])

    max_halving_attempts = 3
    lr_log = []
    halving_log = []

    if checkpoint_path is None:
        checkpoint_path = str(_default_checkpoint_path())

    checkpoint = load_checkpoint_cross(checkpoint_path)
    if checkpoint:
        current_epoch, best_loss = checkpoint["epoch"], checkpoint["best_loss"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for name, param in zip(param_names, params):
            param.data = checkpoint[name].clone().detach().to(device)
        if logging:
            print(f"Resuming cross fitting from epoch {current_epoch}; best loss {best_loss}")
    else:
        current_epoch = 0

    best_params = {name: param.clone() for name, param in zip(param_names, params)}
    loss_histories = []

    for epoch in range(current_epoch, number_of_cycles):
        total_nll = 0.0
        try:
            for batch_idx, (X_batch, Y_batch) in enumerate(zip(X_groups, Y_groups)):
                is_last_batch = batch_idx == len(X_groups) - 1

                for param_group, name in zip(optimizer.param_groups, lr_set.keys()):
                    param_group["lr"] = lr_set[name]

                for step in range(steps_per_batch):
                    alpha_matrix, nu_matrix, sigma_matrix = compute_parameter_matrices(
                        Delta_A, Delta_B, rho_A, rho_B, rho_V, W, alpha, nu, sigma
                    )
                    K = (
                        approx_matern_kernel_cross_legacy(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                        if use_legacy_kernel
                        else approx_matern_kernel_cross(alpha_matrix, nu_matrix, sigma_matrix, X_batch)
                    )

                    nll = negative_log_likelihood(Y_batch, K)
                    if is_last_batch:
                        total_nll += nll.item()

                    optimizer.zero_grad()
                    nll.backward()

                    model_state_before_step = {name: param.clone() for name, param in zip(param_names, params)}
                    grad_state_before_step = {f"{name}_grad": param.grad.clone() for name, param in zip(param_names, params)}

                    _ = perform_optimization_step_with_halving(
                        optimizer,
                        params,
                        param_names,
                        model_state_before_step,
                        grad_state_before_step,
                        max_halving_attempts,
                        epoch,
                        step,
                        halving_log,
                        lr_log,
                        Delta_A,
                        Delta_B,
                        rho_A,
                        rho_B,
                        rho_V,
                        W,
                        alpha,
                        nu,
                        sigma,
                        X_batch,
                    )

        except Exception as exc:
            print(f"Error during epoch {epoch}: {exc}")
            print(f"Parameters -> Delta_A: {Delta_A}, Delta_B: {Delta_B}, rho_A: {rho_A}, rho_B: {rho_B}, rho_V: {rho_V}, W: {W}")
            break

        loss_histories.append(total_nll)
        if total_nll < best_loss - tolerance:
            best_loss = total_nll
            epochs_no_improve = 0
            for name, param in zip(param_names, params):
                best_params[name] = param.clone()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                if print_early_stopping_epochs:
                    print("Cross optimization early stopping at epoch", epoch)
                break

        elapsed_time = time.time() - start_time
        if checkpoint_interval and epoch % checkpoint_interval == 0:
            save_checkpoint_cross(epoch, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, optimizer, best_loss, checkpoint_path)

        if elapsed_time >= max_time_seconds - 600:
            save_checkpoint_cross(epoch, Delta_A, Delta_B, rho_A, rho_B, rho_V, W, optimizer, best_loss, checkpoint_path)
            return params, best_params, loss_histories

    return params, best_params, loss_histories


def optimize_cross_parameters(
    optimized_marginal_params: list,
    X: torch.Tensor,
    Y: torch.Tensor,
    number_of_groups: int,
    number_of_cycles: int = 500,
    steps_per_batch: int = 20,
    print_early_stopping_epochs: bool = False,
    checkpoint_interval: int = 50,
    max_time_hours: float = 24,
    use_legacy_kernel: bool = False,
):
    """Optimize cross parameters by splitting data into groups."""
    n_locations = X.size(0)
    group_size = max(1, n_locations // number_of_groups)
    X_groups = torch.split(X, group_size)
    Y_groups = torch.split(Y, group_size)

    lr_set = {
        "Delta_A": 0.1,
        "Delta_B": 0.1,
        "rho_A": 0.1,
        "rho_B": 0.1,
        "rho_V": 0.1,
        "W": 0.1,
    }

    return optimize_cross_parameters_in_groups(
        optimized_marginal_params,
        lr_set,
        list(X_groups),
        list(Y_groups),
        number_of_cycles=number_of_cycles,
        steps_per_batch=steps_per_batch,
        print_early_stopping_epochs=print_early_stopping_epochs,
        checkpoint_interval=checkpoint_interval,
        max_time_hours=max_time_hours,
        use_legacy_kernel=use_legacy_kernel,
    )
