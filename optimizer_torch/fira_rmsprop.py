# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple

import torch
from torch import nn
from torch.optim import Optimizer

from transformers.utils.versions import require_version

from .gradient_projection import GradientProjector

class RMSProp(Optimizer):
    """
    Implements RMSProp algorithm with optional gradient projection.

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        beta (`float`, *optional*, defaults to 0.99): To distinguish it from galore's alpha hyperparameter.
            Smoothing constant for the moving average of squared gradients.
        eps (`float`, *optional*, defaults to 1e-08):
            Term added to the denominator to improve numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Weight decay to apply.
        momentum (`float`, *optional*, defaults to 0.0):
            The momentum factor.
        centered (`bool`, *optional*, defaults to False):
            If True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance.
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        beta: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
        centered: bool = False,
        no_deprecation_warning: bool = False,
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
        require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"Invalid beta parameter: {beta} - should be in [0.0, 1.0)")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum parameter: {momentum} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "beta": beta, "eps": eps, "weight_decay": weight_decay, "momentum": momentum, "centered": centered}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad

                ##### TODO: weight decay, not covered by current experiments. #####
                # if group["weight_decay"] > 0.0:
                #     grad.add_(p, alpha=group["weight_decay"])

                if grad.is_sparse:
                    raise RuntimeError("Don't support sparse gradients")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                                    
                # Gradient Projection
                if "rank" in group:
                    if "projector" not in state:
                        state["projector"] = GradientProjector(group["rank"], update_proj_gap=group["update_proj_gap"], alpha=group["alpha"], proj_type=group["proj_type"])
                    grad = state["projector"].project(grad, state["step"])

                # State initialization
                if "square_avg" not in state:
                    state["square_avg"] = torch.zeros_like(grad)
                square_avg = state["square_avg"]

                if group["momentum"] > 0.0:
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(grad)
                    momentum_buffer = state["momentum_buffer"]
                else:
                    momentum_buffer = None

                if group["centered"]:
                    if "grad_avg" not in state:
                        state["grad_avg"] = torch.zeros_like(grad)
                    grad_avg = state["grad_avg"]
                else:
                    grad_avg = None

                state["step"] += 1

                beta = group["beta"]

                # Update moving averages of squared gradients
                square_avg.mul_(beta).addcmul_(grad, grad, value=1 - beta)

                if group["centered"]:
                    grad_avg.mul_(beta).add_(grad, alpha=1 - beta)
                    avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt().add_(group["eps"])
                else:
                    avg = square_avg.sqrt().add_(group["eps"])

                # Apply momentum
                if group["momentum"] > 0.0:
                    momentum_buffer.mul_(group["momentum"]).addcdiv_(grad, avg)
                    norm_grad = momentum_buffer
                else:
                    norm_grad = grad / avg
                                    
                # Gradient Projection Back
                if "rank" in group:

                    # Norm-Based Scaling
                    subgrad = state["projector"].project_back(grad)
                    norm_dim = 0 if norm_grad.shape[0] < norm_grad.shape[1] else 1
                    scaling_factor = (
                        torch.norm(norm_grad, dim=norm_dim) /
                        (torch.norm(grad, dim=norm_dim) + 1e-8)
                    )
                    if norm_dim == 1:
                        scaling_factor = scaling_factor.unsqueeze(1)
                    scaling_grad = (p.grad - subgrad) * scaling_factor

                    # Norm-Growth Limiter
                    if "scaling_grad" in state:
                        scaling_grad_norm = torch.norm(scaling_grad)
                        limiter = max(
                                scaling_grad_norm / 
                                (state["scaling_grad"] + 1e-8),
                                1.01,
                            ) / 1.01
                        scaling_grad = scaling_grad / limiter
                        state["scaling_grad"] = scaling_grad_norm / limiter
                    else:
                        state["scaling_grad"] = torch.norm(scaling_grad)
                    
                    norm_grad = state["projector"].project_back(norm_grad) + scaling_grad

                step_size = group["lr"]
                p.add_(norm_grad, alpha=-step_size)

        return loss