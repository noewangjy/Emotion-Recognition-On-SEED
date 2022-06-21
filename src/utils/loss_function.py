import torch
from torch import Tensor as T
import torch.nn as nn
from typing import List, Callable, Tuple


# def mmd_loss_function(num_class: int, sigmas: List[float] = None) -> Callable:
#     """
#     Computes the Maximum Mean Discrepancy loss between the distributions of source and target.
#     Args:
#       num_class: an int that is the number of source samples
#       sigmas: a list of floats that specifies the variances of the RBF kernels.
#     Returns:
#       a scalar representing the mean-median kernel distance between source
#       and target.
#     """
#     if sigmas is None:
#         sigmas = torch.tensor([1, 5, 10], dtype=torch.float)
#     else:
#         sigmas = torch.tensor(sigmas, dtype=torch.float)
#
#     def compute_loss(
#             source_pred: T,
#             target_pred: T,
#             source_label: T,
#             target_label: T
#     ) -> T:
#         """Computes the Maximum Mean Discrepancy loss between the distributions
#         of source and target.
#         Args:
#             source_pred: a tensor of shape [num_source_samples, num_classes]
#                 containing the predicted labels for the source samples.
#             target_pred: a tensor of shape [num_target_samples, num_classes]
#                 containing the predicted labels for the target samples.
#             source_label: a tensor of shape [num_source_samples] containing
#                 the true labels for the source samples.
#             target_label: a tensor of shape [num_target_samples] containing
#                 the true labels for the target samples.
#         Returns:
#             a scalar representing the mean-median kernel distance between source
#             and target.
#         """
#
#         cost: T = torch.tensor(0.0).to(source_pred.device)
#         for i in range(num_class):
#             source_i = source_pred[source_label == i]
#             target_i = target_pred[target_label == i]
#             cost += mmd_two_distributions(source_i, target_i, sigmas)
#
#         return cost / num_class
#
#     return compute_loss
#
#
# def mmd_two_distributions(source: T, target: T, sigmas: T) -> T:
#     """Computes the Maximum Mean Discrepancy loss between the distributions
#     source and target.
#     Args:
#       source: a tensor of shape [num_source_samples, num_features]
#       target: a tensor of shape [num_target_samples, num_features]
#       sigmas: a tensor of shape [num_sigmas]
#     Returns:
#       a tensor representing the Maximum discrepancy distance between source
#       and target.
#     """
#     # Compute the maximum mean discrepancy.
#     source_kernel = rbf_kernel(source, source, sigmas)
#     target_kernel = rbf_kernel(target, target, sigmas)
#     source_target_kernel = rbf_kernel(source, target, sigmas)
#     mmd_loss = source_kernel + target_kernel - 2. * source_target_kernel
#     return mmd_loss
#
#
# def rbf_kernel(x: T, y: T, sigmas: T) -> T:
#     """Computes the rbf (gaussian) kernel between x and y.
#     Args:
#       x: a tensor of shape [num_x_samples, num_features]
#       y: a tensor of shape [num_y_samples, num_features]
#       sigmas: a tensor of shape [num_sigma] the variance of the rbf kernel
#     Returns:
#         A tensor of shape [] containing the rbf kernel between x and y for all samples.
#
#     """
#     sigmas = sigmas.unsqueeze(1).to(x.device)
#     beta = 1. / (2. * sigmas)
#     samples_pairwise_distances = compute_pairwise_distances(x, y)
#     rbf_kernel_output = torch.mean(torch.exp(-torch.matmul(beta, torch.reshape(samples_pairwise_distances, (1, -1)))))
#     return rbf_kernel_output
#
#
# def compute_pairwise_distances(x: T, y: T) -> T:
#     """Computes the squared pairwise Euclidean distances between x and y.
#     Args:
#       x: a tensor of shape [num_x_samples, num_features]
#       y: a tensor of shape [num_y_samples, num_features]
#     Returns:
#       a distance matrix of dimensions [num_x_samples, num_y_samples].
#     """
#     # shape: [num_x_samples, 1, num_features]
#     expanded_x = torch.unsqueeze(x, 1)
#     # shape: [1, num_y_samples, num_features]
#     expanded_y = torch.unsqueeze(y, 0)
#     # shape: [num_x_samples, num_y_samples]
#     return ((expanded_x - expanded_y) ** 2).sum(dim=-1)

###############################
# Version 2
###############################
#
#
# def _mix_rbf_kernel(x: T, y: T, sigma_list: List[float]) -> Tuple[T, T, T, int]:
#     """Computes the mixed rbf (gaussian) kernel between x and y.
#     Args:
#       x: a tensor of shape [num_x_samples, num_features]
#       y: a tensor of shape [num_y_samples, num_features]
#       sigma_list: a list of floats that specifies the variances of the rbf kernels.
#     Returns:
#       A tuple (gamma, delta, mixed_rbf_kernel, num_sigmas).
#     """
#     # assert x.size(0) == y.size(0)
#     num_samples = x.size(0)
#     z = torch.cat([x, y], dim=0)
#     zzt = torch.matmul(z, z.t())
#     diag_zzt = torch.diag(zzt).unsqueeze(1)
#     z_norm_squared = diag_zzt.expand_as(zzt)
#     exponent = z_norm_squared - 2 * zzt + z_norm_squared.t()
#
#     # compute the mixed rbf kernel
#     kernel: T = torch.zeros_like(exponent).to(x.device)
#     for sigma in sigma_list:
#         gamma = 1.0 / (2 * sigma ** 2)
#         kernel += torch.exp(-gamma * exponent)
#
#     return kernel[:num_samples, :num_samples], kernel[:num_samples, num_samples:], kernel[num_samples:,
#                                                                                    num_samples:], len(sigma_list)
#
#
# def _mmd2(kernel_xx: T, kernel_xy: T, kernel_yy: T, constant_diagonal: T = None, biased: bool = False) -> T:
#     """Computes the squared maximum mean discrepancy (MMD) of two matrices
#     Args:
#       kernel_xx: the squared kernel matrix associated with X
#       kernel_xy: the squared kernel matrix associated with X and Y
#       kernel_yy: the squared kernel matrix associated with Y
#       constant_diagonal: whether to use a constant diagonal in the computation
#       biased: whether to use a biased estimate or not
#     Returns:
#       The maximum mean discrepancy estimate
#     """
#     num_x_samples = kernel_xx.size(0)
#     num_y_samples = kernel_yy.size(0)
#
#     if constant_diagonal:
#         diag_x = diag_y = constant_diagonal
#         sum_diag_x = num_x_samples * constant_diagonal
#         sum_diag_y = num_y_samples * constant_diagonal
#     else:
#         diag_x = torch.diag(kernel_xx)
#         diag_y = torch.diag(kernel_yy)
#         sum_diag_x = torch.sum(diag_x)
#         sum_diag_y = torch.sum(diag_y)
#
#     kt_xx_sum = (kernel_xx.sum(dim=1) - diag_x).sum()
#     kt_yy_sum = (kernel_yy.sum(dim=1) - diag_y).sum()
#     kt_xy_sum = kernel_xy.sum()
#
#     if biased:
#         mmd2 = (kt_xx_sum + sum_diag_x) / num_x_samples ** 2 + \
#                (kt_yy_sum + sum_diag_y) / num_y_samples ** 2 - \
#                2.0 * kt_xy_sum / (num_x_samples * num_y_samples)
#     else:
#         mmd2 = kt_xx_sum / (num_x_samples * (num_x_samples - 1)) + \
#                kt_yy_sum / (num_y_samples * (num_y_samples - 1)) - \
#                2.0 * kt_xy_sum / (num_x_samples * num_y_samples)
#
#     return mmd2
#
#
# def mmd_loss(source: T, target: T, sigma_list: List[float], biased: bool = False) -> T:
#     """Computes the maximum mean discrepancy (MMD) of two matrices.
#     Args:
#       source: a tensor of shape [num_source_samples, num_features]
#       target: a tensor of shape [num_target_samples, num_features]
#       sigma_list: a list of floats that specifies the variances of the rbf kernels.
#       biased: whether to use a biased estimate or not
#     Returns:
#       The maximum mean discrepancy estimate
#     """
#     num_features = source.size(1)
#     kernel_xx, kernel_xy, kernel_yy, num_sigmas = _mix_rbf_kernel(source, target, sigma_list)
#     return _mmd2(kernel_xx, kernel_xy, kernel_yy, biased=biased) / num_features
#

########################################################
# Version 3: pytorch_DAN
########################################################

def compute_pairwise_distances(x: T, y: T) -> T:
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
    """
    assert x.size(1) == y.size(1), ValueError('x and y should have the same feature_dimension')
    # shape: [num_x_samples, 1, num_features]
    expanded_x = torch.unsqueeze(x, 1)
    # shape: [1, num_y_samples, num_features]
    expanded_y = torch.unsqueeze(y, 0)
    # shape: [num_x_samples, num_y_samples]
    return torch.sum((expanded_x - expanded_y) ** 2, dim=-1)


def compute_kernel_matrix(x: T, y: T, sigma_list: List[float]) -> T:
    """Computes the squared pairwise Euclidean distances between x and y.
    Args:
      x: a tensor of shape [num_x_samples, num_features]
      y: a tensor of shape [num_y_samples, num_features]
      sigma_list: a list of floats that specifies the variances of the rbf kernels.
    Returns:
      a distance matrix of dimensions [num_x_samples, num_y_samples].
        """

    sigmas: T = torch.tensor(sigma_list, dtype=torch.float).to(x.device)
    sigmas = sigmas.view(-1, 1)
    beta = 1. / (2. * sigmas)
    distance = compute_pairwise_distances(x, y).contiguous()
    distance_ = distance.view(1, -1)
    s = torch.matmul(beta, distance_)
    s = torch.exp(-s)

    return torch.sum(s, dim=0, keepdim=True)


def mmd_two_distributions(source: T, target: T, sigma_list: List[float]) -> T:
    """Computes the maximum mean discrepancy (MMD) of two matrices.
    Args:
        source: a tensor of shape [num_source_samples, num_features]
        target: a tensor of shape [num_target_samples, num_features]
        sigma_list: a list of floats that specifies the variances of the rbf kernels.
    Returns:
        The maximum mean discrepancy estimate
    """
    xx = compute_kernel_matrix(source, source, sigma_list)
    yy = compute_kernel_matrix(target, target, sigma_list)
    xy = compute_kernel_matrix(source, target, sigma_list)
    return xx + yy - 2 * xy


def mmd_loss(source_features: T, target_features: T, sigma_list: List[float]) -> T:
    """Computes the maximum mean discrepancy (MMD) of two matrices.
    Args:
      source_features: a tensor of shape [num_source_samples, num_features]
      target_features: a tensor of shape [num_target_samples, num_features]
      sigma_list: a list of floats that specifies the variances of the rbf kernels.
    Returns:
      The maximum mean discrepancy estimate
    """
    ss_loss: T = torch.mean(compute_kernel_matrix(source_features, source_features, sigma_list))
    st_loss: T = torch.mean(compute_kernel_matrix(source_features, target_features, sigma_list))
    tt_loss: T = torch.mean(compute_kernel_matrix(target_features, target_features, sigma_list))
    loss: T = ss_loss + tt_loss - 2.0 * st_loss
    return loss.sqrt()


class MMD_AAE_Loss(object):
    def __init__(self, num_sources: int, sigma_list: List[float]):
        self.num_sources: int = num_sources
        self.sigma_list: List[float] = sigma_list

    def calculate(self, y_pred: T) -> T:
        """Computes the maximum mean discrepancy (MMD) of two matrices.
        Args:
          y_pred: a tensor of shape [num_sources, num_samples, num_features]
        Returns:
          The maximum mean discrepancy estimate
        """

        losses: List[T] = []
        for i in range(self.num_sources):
            domain_i = y_pred[i, :, :]
            for j in range(i + 1, self.num_sources):
                domain_j = y_pred[j, :, :]
                losses.append(mmd_loss(domain_i, domain_j, self.sigma_list))
        return torch.stack(losses).mean()
