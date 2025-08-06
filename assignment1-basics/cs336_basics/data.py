import torch 
import numpy.typing as npt

def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: torch.device | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a batch of inputs and targets from the dataset.

    Args:
        dataset (npt.NDArray): The dataset to sample from.
        batch_size (int): The number of samples in the batch.
        context_length (int): The length of each input sequence.
        device (torch.device | None): The device to place tensors on.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing the input sequences and their corresponding targets.
    """
    dataset_length = len(dataset)
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long)

    for i in range(batch_size):
        start_index = torch.randint(0, dataset_length - context_length, (1,)).item()
        input_seq = dataset[start_index : start_index + context_length]
        target_seq = dataset[start_index + 1 : start_index + context_length + 1]
        inputs[i] = torch.tensor(input_seq, dtype=torch.long)
        targets[i] = torch.tensor(target_seq, dtype=torch.long)

    if device:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
    return inputs, targets