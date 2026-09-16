from __future__ import annotations

from collections.abc import Callable
from typing import Literal, cast

import torch


OdeSchedule = Literal["uniform", "cosine"]
SUPPORTED_ODE_SCHEDULES = ("uniform", "cosine")


def validate_ode_schedule(schedule: str) -> OdeSchedule:
    if schedule not in SUPPORTED_ODE_SCHEDULES:
        raise ValueError("schedule must be 'uniform' or 'cosine'")
    return cast(OdeSchedule, schedule)


def euler_sample_patch(
    velocity_function: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    history: torch.Tensor,
    noise: torch.Tensor,
    *,
    steps: int = 25,
    schedule: OdeSchedule = "uniform",
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    schedule = validate_ode_schedule(schedule)
    current = noise.clone()
    if schedule == "uniform":
        delta = 1.0 / steps
        for step in range(steps):
            time = torch.full(
                (current.shape[0],),
                step / steps,
                device=current.device,
                dtype=current.dtype,
            )
            current = current + delta * velocity_function(history, current, time)
        return current

    integration_dtype = (
        torch.float32
        if current.dtype in (torch.float16, torch.bfloat16)
        else current.dtype
    )
    time_grid = torch.linspace(
        0.0,
        1.0,
        steps + 1,
        device=current.device,
        dtype=integration_dtype,
    )
    time_grid = 1.0 - torch.cos(time_grid * (torch.pi / 2.0))
    for step in range(steps):
        time = time_grid[step].to(dtype=current.dtype).expand(current.shape[0])
        delta = (time_grid[step + 1] - time_grid[step]).to(dtype=current.dtype)
        current = current + delta * velocity_function(history, current, time)
    return current


def cfg_combine(unconditional: torch.Tensor, conditional: torch.Tensor, scale: float) -> torch.Tensor:
    return conditional + scale * (conditional - unconditional)
