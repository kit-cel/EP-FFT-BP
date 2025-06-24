# Copyright 2024, 2025 Luca Schmid.
import torch as t
from scipy.linalg import dft

class DFT:
    """
    Class implementing the discrete Fourier transform (DFT) in the context of Gaussian random variables.
    This code is part of the publication [1].

    [1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
    International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC),
    Surrey, UK, July 2025.
    """
    def __init__(self, N: int, inverse: bool = False, complex_interleaved: bool = False, dev: t.device = t.device('cpu'), precision: str = 'double'):
        """
        :param N: DFT size
        :param inverse: Flag, if DFT or inverse DFT.
        :param complex_interleaved: Flag, if real/imaginary part of each complex value are stacked next to each other (as in [1]) or blockwise.
        :param dev: cuda device
        :param precision: 'float' or 'double'
        """
        self.dev = dev
        self.N = N
        self.inverse = inverse
        self.complex_interleaved = complex_interleaved
        scale = 'n' if self.inverse else None

        self.block_diag_row_idxs, self.block_diag_cols_idxs = self._get_block_diagonal_idx(N=N)

        if precision == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif precision == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {precision}.")


        W = t.from_numpy(dft(N, scale=scale)).to(self.cprec) # Vandermonde  matrix
        self.dft_matrix = t.conj(W) if self.inverse else W
        if complex_interleaved:
            self.dft_matrix_eqiv_real = t.zeros(size=(2*N,2*N), dtype=self.prec)
            self.dft_matrix_eqiv_real[ ::2, ::2] =  self.dft_matrix.real
            self.dft_matrix_eqiv_real[ ::2,1::2] = -self.dft_matrix.imag
            self.dft_matrix_eqiv_real[1::2, ::2] =  self.dft_matrix.imag
            self.dft_matrix_eqiv_real[1::2,1::2] =  self.dft_matrix.real
        else:
            self.dft_matrix_eqiv_real = t.cat((t.cat((self.dft_matrix.real, -self.dft_matrix.imag), dim=-1), t.cat((self.dft_matrix.imag, self.dft_matrix.real), dim=-1)), dim=-2)

        self.inf = 1e12

    def _get_block_diagonal_idx(self, N):
        """ Helper function to generate indices for rows/columns of the 2x2 block diagonal elements."""
        row_idxs = t.arange(2*N).repeat_interleave(2)
        cols_idxs = (t.tensor([0,1], device=self.dev).unsqueeze(0) + (2*t.arange(N).repeat_interleave(2)).unsqueeze(-1)).flatten()
        return row_idxs, cols_idxs

    def deterministic_forward(self, x):
        """
        Apply DFT/IDFT matrix to input x.
        :param x: Input tensor of shape (...,self.N).
        :return: Transformed tensor of schape (...,self.N).
        """
        if x.dim() == 1: # x is a single vector
            assert x.shape == (self.N,)
            return t.matmul(self.dft_matrix, x)
        elif x.dim() == 2: # x is a batch of vectors of shape (batch_size, N)
            batch_size = x.shape[0]
            assert x.shape[1] == self.N
            return t.bmm(self.dft_matrix.unsqueeze(0), x.view(batch_size,self.N,1)).view(batch_size, self.N)
        else:
            assert False

    def gaussian_forward(self, mean, cov):
        """
        Apply linear DFT transform to Gaussian random variable with given mean and covariance matrix.
        :param mean: Mean tensors of shape (...,self.N,2) or (...,2*self.N). Factor 2 is real/imag part.
        :param cov: Covariance matrices of shape (...,self.N,2,2) or (...,2*self.N,2*self.N).
        :return: Transformed mean/cov of shape (...,2*self.N)/(...,2*self.N,2*self.N).
        """
        if mean.shape[-1] == 2:
            assert self.complex_interleaved
            assert cov.shape[-2:] == (2,2)
            mean = mean.view(mean.shape[:-2]+[2*self.N,])
            cov = t.zeros(size=cov.shape[:-3]+[2*self.N,2*self.N], dtype=self.prec, device=self.dev)
            cov = t.diag_embed(t.diagonal(cov, dim1=-2, dim2=-1).view(-1, 2*self.N))
            cov = cov.view(cov.shape[:-3]+[])

        assert mean.shape[-1] == 2*self.N
        assert mean.shape[-1] == cov.shape[-2]
        assert mean.shape[-1] == cov.shape[-1]
        assert len(cov.shape) == len(mean.shape)+1
        if mean.dim() == 1: # single vector
            new_mean = t.matmul(self.dft_matrix_eqiv_real, mean)
            new_cov = t.matmul(t.matmul(self.dft_matrix_eqiv_real, cov),self.dft_matrix_eqiv_real.transpose(0,1))
        elif mean.dim() == 2: # batch of elements
            batch_size = mean.shape[0]
            new_mean = t.bmm(self.dft_matrix_eqiv_real.unsqueeze(0).repeat(batch_size,1,1), mean.view(batch_size,2*self.N,1)).view(batch_size, 2*self.N)
            new_cov = t.bmm(t.bmm(self.dft_matrix_eqiv_real.unsqueeze(0).repeat(batch_size,1,1), cov),self.dft_matrix_eqiv_real.transpose(0,1).unsqueeze(0).repeat(batch_size,1,1))
        return new_mean, new_cov

    def block_diag2full_matrix(self, block_diag):
        """
        Embed N 2x2 matrices in 2Nx2N block diagonal matrix.
        """
        assert len(block_diag.shape) == 4
        batch_size = block_diag.shape[0]
        assert block_diag.shape == (batch_size, self.N, 2, 2)
        assert self.complex_interleaved

        # build full complex interleaved (2N x 2N) covariance matrix
        full_cov = t.zeros((batch_size, 2*self.N, 2*self.N), dtype=self.prec, device=self.dev)
        full_cov[:, self.block_diag_row_idxs, self.block_diag_cols_idxs] = block_diag.flatten(start_dim=1)
        return full_cov

    def full_matrix2_block_diag(self, full_matrix):
        """
        Extract N 2x2 matrices from 2Nx2N block diagonal matrix.
        """
        assert len(full_matrix.shape) == 3
        batch_size = full_matrix.shape[0]
        assert full_matrix.shape == (batch_size, 2*self.N, 2*self.N)

        # extract block diag elements and change view
        block_diag = full_matrix[:, self.block_diag_row_idxs, self.block_diag_cols_idxs].view(batch_size, self.N, 2, 2)
        return block_diag
