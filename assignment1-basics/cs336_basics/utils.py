import torch
import torch.nn as nn


def save_checkpoint(model, optimizer, iteration, out):
    """
    Save the model and optimizer state to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        iteration (int): The current iteration number.
        out (str): The output file path for the checkpoint.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(src, model, optimizer):
    """
    Load the model and optimizer state from a checkpoint file.

    Args:
        model (nn.Module): The model to load.
        optimizer (torch.optim.Optimizer): The optimizer to load.
        in_path (str): The input file path for the checkpoint.
    """
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    iteration = checkpoint["iteration"]
    return iteration
