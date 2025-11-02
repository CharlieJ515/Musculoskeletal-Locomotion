import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from .writer import writer


def log_weight_hist(tag: str, model: nn.Module, step: int):
    for name, param in model.named_parameters():
        writer.add_histogram(f"{tag}/{name}", param, step)


def log_grad_hist(tag: str, model: nn.Module, step: int):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        writer.add_histogram(f"{tag}/{name}", param.grad, step)
