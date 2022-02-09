import time
import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt


def randcn(*args, **kwargs):
    """
    Use similarly to torch.randn, but returns complex normal and normalized by 1/sqrt(2).
    """
    re = torch.randn(*args, **kwargs) / torch.tensor(2).sqrt()
    im = torch.randn(*args, **kwargs) / torch.tensor(2).sqrt()
    return torch.complex(re, im)


def randc(*args, **kwargs):
    """
    Use similarly to torch.rand, but returns complex numbers from within [-1,1]x[-1j,1j].
    """
    re = 2 * torch.rand(*args, **kwargs) - 1
    im = 2 * torch.rand(*args, **kwargs) - 1
    return torch.complex(re, im)


def logdet(input):
    """
    Returns the real (due to numeric problems) part of the logdet of the input.
    :param input: invertible matrix
    :return: log-determinant with autograd
    """
    _, logdetval = torch.linalg.slogdet(input)
    # return LogDet.apply(input).real
    return logdetval


def cmat_square(input):
    """
    Calculates input * input^H and cleans up diagonal
    :param input: complex matrix
    :return: matrix square
    """
    square = torch.matmul(input, input.conj().transpose(-2, -1))
    return clean_hermitian(square)


def clean_hermitian(input):
    """
    Cleans up the diagonal of a hermitian matrix from numeric errors.
    :param input: matrix or stack of matrices
    :return: cleaned up matrices
    """
    """
    output = input.clone()
    output.diagonal(dim1=-2, dim2=-1).imag = torch.zeros(1)
    """
    tempcopy = input.clone()
    output = (tempcopy + tempcopy.conj().transpose(-2, -1)) / 2
    output.diagonal(dim1=-2, dim2=-1).imag = torch.zeros(1)
    return output


def mmchain(*args, **kwargs):
    """
    Calculates the chain of matrix products of two or more matrix batches
    :param args: two or more matrix batches
    :return: matrix product
    """

    if len(args) >= 2:
        return torch.matmul(args[0], mmchain(*args[1:]), **kwargs)
    else:
        return args[0]


def btrace(input):
    """
    Calculates the trace of a matrix batch
    :param input: batch of quadratic matrices
    :return: batch of the traces of matrices
    """
    return input.diagonal(dim1=-2, dim2=-1).sum(-1)


def bf_mat_pow(bf_mat):
    """
    Calulates the power of a beamformer matrices.
    :param bf_mat: beamformer matrix or batch of beamformer matrices
    :return: power as (*batch_size)
    """
    power = torch.square(bf_mat.real) + torch.square(bf_mat.imag)
    return power.sum(dim=-2).sum(dim=-1)


def complex_relu(input):
    """
    Performs ReLU operation on complex tensor by rectifying real and imag part individually.
    :param input: tensor
    :return: tensor
    """
    return torch.complex(F.relu(input.real), F.relu(input.imag))


def complex_mod_relu(input, bias):
    """
    Performs the modReLU operation according to Trabelsi et al. 'Deep Complex Networks'.
    :param input: (*batch_size, N) array
    :param bias: (N) array of real numbers
    :return: (*batch_size, N) array
    """
    mag = torch.abs(input) + bias
    zeropos = mag < 0
    output = mag * torch.sgn(input)
    output[zeropos] = 0
    return output


class rationalfct_solve_0d2_cls(torch.autograd.Function):
    @staticmethod
    def forward(ctx, nom, denom, num_iter, x_init):
        batch_size = list(nom.size())[1:]

        x_root = torch.full([1, *batch_size], x_init, dtype=nom.dtype, device=nom.device)
        for iter in range(num_iter):
            x_root = rationalfct_solve_0d2_iteration(x_root, nom, denom)

        ctx.save_for_backward(nom, denom, x_root)

        x_root = x_root.squeeze(0)
        return x_root

    @staticmethod
    def backward(ctx, grad_xout):
        nom, denom, x_root = ctx.saved_tensors
        grad_xout = grad_xout.unsqueeze(0)
        grad_nom = grad_denom = grad_num_iter = grad_x_init = None
        nz_pos = x_root != 0
        denom_sum = torch.clamp(denom + x_root, min=1e-32)  # safety step to avoid div by 0 at all costs
        ratfun_grad_mu = -2 * torch.nansum(nom / denom_sum**3, dim=0, keepdim=True)

        if ctx.needs_input_grad[0]:
            grad_nom = -1 / ratfun_grad_mu / denom_sum ** 2
            grad_nom = grad_nom * grad_xout
            grad_nom = grad_nom * nz_pos
        if ctx.needs_input_grad[1]:
            grad_denom = 2 / ratfun_grad_mu * nom / denom_sum ** 3
            grad_denom = grad_denom * grad_xout
            grad_denom = grad_denom * nz_pos

        return grad_nom, grad_denom, grad_num_iter, grad_x_init


def rationalfct_solve_0d2(nom, denom, num_iter=4, x_init=1e-12):
    return rationalfct_solve_0d2_cls.apply(nom, denom, num_iter, x_init)


def rationalfct_solve_0d2_iteration(xin, nom, denom):
    denom_sum = torch.clamp(denom + xin, min=1e-32)  # safety step to avoid div by 0 at all costs
    fracsum_cube = torch.nansum(nom / denom_sum / denom_sum / denom_sum, dim=0, keepdim=True)
    fracsum_square = torch.nansum(nom / denom_sum / denom_sum, dim=0, keepdim=True)

    big_nom = fracsum_square.pow(1.5) - fracsum_square
    big_frac = big_nom / fracsum_cube

    xtemp = xin + big_frac
    xout = torch.clamp(xtemp, min=0)

    return xout


def matpoly_simple_norm(mat, poly_coeff, norm_above=1):
    """
    Calculates the matrix polynom with optionally multiple features
    :param mat: tensor of quadratic matrices (*batch_size, M, M)
    :param poly_coeff: tensor of coefficients belonging to poly_exp (len(polyexp))
    :return (*batch_size, M, M)
    """
    n = len(poly_coeff)  # degree + 1

    batch_size = list(mat.size())[:-2]
    expand_dim = [1] * len(batch_size)
    mat_size = list(mat.size())[-2:]
    device = mat.device
    dtype = mat.dtype
    norm = btrace(mat) / mat_size[0]

    degrees = []
    eye = torch.eye(mat_size[0], dtype=dtype, device=device).expand(*batch_size, *mat_size)  # (*bs, M, M)
    if norm_above == -1:
        # degrees.append(eye * norm.unsqueeze(-1).unsqueeze(-1) / 10)  # ADDED DIVIDE BY 10 FOR TESTING REASONS
        degrees.append(eye * norm.unsqueeze(-1).unsqueeze(-1))
        norm_above = 1
    else:
        degrees.append(eye)
    running = eye
    for i in range(1, n):
        if i > norm_above:
            running = torch.matmul(running, mat / norm.unsqueeze(-1).unsqueeze(-1))
        else:
            running = torch.matmul(running, mat)
        degrees.append(running)
    degrees = torch.stack(degrees, dim=0)

    poly_coeff_size = list(poly_coeff.size())
    result = torch.sum(poly_coeff.view(*poly_coeff_size, *expand_dim, 1, 1) * degrees, dim=0)  # sum polys

    return result