import torch
def get_value_from_distribution(dist, min_val, max_val):
    """
    Convert a distribution into a value by taking the expected value.

    Parameters:
    dist (torch.Tensor): (batch, distribution_size, distribution_bins) The probability distribution.
    bins (int): The number of bins in the distribution.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.

    Returns:
    torch.Tensor: The estimated continuous values of shape (batch, distribution_size).
    """
    batch, dist_length, dist_bins = dist.shape
    dist = dist.view(batch * dist_length, dist_bins)
    bin_values = torch.linspace(min_val, max_val, dist_bins, device=dist.device).view(1, dist_bins)
    values = dist * bin_values
    values = values.sum(dim=-1)
    return values


def get_twohot_from_value(value, dist_bins, min_val, max_val):
    """
    Compute the two-hot encoding for a given value based on evenly spaced bins.

    Parameters:
    value (torch.Tensor): (batch, distribution_size) The input values to encode.
    dist_bins (int): The number of bins to discretize the range.
    min_val (float): The minimum value of the range.
    max_val (float): The maximum value of the range.

    Returns:
    torch.Tensor: Two-hot encoded distribution of shape (batch, distribution_size, dist_bins).
    """
    batch, dist_length = value.shape
    bin_edges = torch.linspace(min_val, max_val, dist_bins, device=value.device)  # (bins,)

    # Find the bin indices
    bin_indices = torch.sum(value.unsqueeze(-1) >= bin_edges, dim=-1) - 1
    bin_indices = torch.clamp(bin_indices, 0, dist_bins - 2)  # Ensure indices are in valid range

    # Compute weights for interpolation
    bin_left = bin_edges[bin_indices]
    bin_right = bin_edges[bin_indices + 1]
    right_weight = (value - bin_left) / (bin_right - bin_left)
    left_weight = 1 - right_weight

    # Create two-hot distribution
    dist = torch.zeros(batch, dist_length, dist_bins, device=value.device)
    dist.scatter_(-1, bin_indices.unsqueeze(-1), left_weight.unsqueeze(-1))
    dist.scatter_(-1, (bin_indices + 1).unsqueeze(-1), right_weight.unsqueeze(-1))

    return dist