"""coil 域指标:与前向模型同域的复数多线圈误差(spec §4.1 主要终点)。"""
from __future__ import annotations

import torch

_AX = (-3, -2, -1)


def fft3c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(x, dim=_AX), dim=_AX), dim=_AX)


def ifft3c(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(x, dim=_AX), dim=_AX), dim=_AX)


def e_coil_full(x: torch.Tensor, csm: torch.Tensor, y_full: torch.Tensor) -> float:
    """||F S x - y_full|| / ||y_full||,x 为复数体积,csm/y_full 线圈轴在前。"""
    num = torch.linalg.vector_norm(fft3c(csm * x.unsqueeze(0)) - y_full)
    return float(num / torch.linalg.vector_norm(y_full))


def e_unobs(x: torch.Tensor, csm: torch.Tensor, y_full: torch.Tensor, mask: torch.Tensor) -> float:
    """未采样区相对误差:||(1-M)(F S x - y_full)|| / ||(1-M) y_full||。"""
    inv = (1.0 - mask).unsqueeze(0)
    num = torch.linalg.vector_norm(inv * (fft3c(csm * x.unsqueeze(0)) - y_full))
    return float(num / torch.linalg.vector_norm(inv * y_full))
