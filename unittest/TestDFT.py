# Copyright 2024, 2025 Luca Schmid.
import unittest
import torch as t
from DFT import DFT

class TestDFT(unittest.TestCase):

    def test_determinsitic_DFT(self, N=2, n_trials=1):
        """
        Tests the deterministic propagation through the DFT for a fixed DFT size N.
        Must produce the same output like t.fft.fft().
        :param N: DFT size.
        :param n_trials: Number of random input vectors to test.
        """
        t.manual_seed(0)
        dft = DFT(N=N, precision='float')
        for trial_i in range(n_trials):
            x_time = t.randn(size=(N,), dtype=t.cfloat)
            assert t.allclose(dft.deterministic_forward(x_time), t.fft.fft(x_time), atol=1e-04), f"Deterministic DFT with N={N} failed (comparison to torch.fft.fft())."

    def test_deterministic_IDFT(self, N=2, n_trials=1):
        """
        Tests the deterministic propagation through the IDFT for a fixed FFT size N.
        Must produce the same output like t.fft.ifft().
        :param N: DFT size.
        :param n_trials: Number of random input vectors to test.
        """
        t.manual_seed(0)
        dft = DFT(N=N, inverse=True, precision='float')
        for trial_i in range(n_trials):
            x_freq = t.randn(size=(N,), dtype=t.cfloat)
            assert t.allclose(dft.deterministic_forward(x_freq), t.fft.ifft(x_freq), atol=1e-04), f"Deterministic IDFT with N={N} failed (comparison to torch.fft.ifft())."

    def test_deterministic_DFT_sweepN(self):
        """
        Tests the deterministic propagation through the DFT/IDFT for different DFT sizes N=[4, 8, 16, 32, 128, 512, 1024].
        Must produce the same output like t.fft.fft() / t.fft.ifft().
        """
        for N in [4, 8, 16, 32, 128, 512, 1024]:
            self.test_determinsitic_DFT(N=N, n_trials=2) # DFT
            self.test_deterministic_IDFT(N=N, n_trials=2) # IDFT

    def test_gaussian_forward_zero_variance(self):
        # N=2
        # real-valued and zero-variance
        dft = DFT(N=2, precision='float')
        mean_real = t.tensor([[1.,2,0,0],[0,-1,0,0]])
        cov = t.zeros(size=(2,4,4))
        mean_out_real, cov_out = dft.gaussian_forward(mean=mean_real, cov=cov)
        fft_out = t.fft.fft(mean_real[:,:2] + 1j*mean_real[:,2:])
        assert t.allclose(mean_out_real, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)
        # imag-valued and zero-variance
        mean_imag = t.tensor([[0, 0, -1, -0.1], [0, 0, 5, 100.1]])
        cov = t.zeros(size=(2, 4, 4))
        mean_out_imag, cov_out = dft.gaussian_forward(mean=mean_imag, cov=cov)
        fft_out = t.fft.fft(mean_imag[:, :2] + 1j * mean_imag[:, 2:])
        assert t.allclose(mean_out_imag, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)
        # complex-valued and zero-variance
        mean_complex = t.tensor([[1,2,3,4], [-4,3.1,0.0,-4.4]])
        cov = t.zeros(size=(2, 4, 4))
        mean_out_complex, cov_out = dft.gaussian_forward(mean=mean_complex, cov=cov)
        fft_out = t.fft.fft(mean_complex[:, :2] + 1j * mean_complex[:, 2:])
        assert t.allclose(mean_out_complex, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)

    def test_gaussian_backward_zero_variance(self):
        # N=2
        # real-valued and zero-variance
        dft = DFT(N=2, inverse=True, precision='float')
        mean_real = t.tensor([[1.,2,0,0],[0,-1,0,0]])
        cov = t.zeros(size=(2,4,4))
        mean_out_real, cov_out = dft.gaussian_forward(mean=mean_real, cov=cov)
        fft_out = t.fft.ifft(mean_real[:,:2] + 1j*mean_real[:,2:])
        assert t.allclose(mean_out_real, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)
        # imag-valued and zero-variance
        mean_imag = t.tensor([[0, 0, -1, -0.1], [0, 0, 5, 100.1]])
        cov = t.zeros(size=(2, 4, 4))
        mean_out_imag, cov_out = dft.gaussian_forward(mean=mean_imag, cov=cov)
        fft_out = t.fft.ifft(mean_imag[:, :2] + 1j * mean_imag[:, 2:])
        assert t.allclose(mean_out_imag, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)
        # complex-valued and zero-variance
        mean_complex = t.tensor([[1,2,3,4], [-4,3.1,0.0,-4.4]])
        cov = t.zeros(size=(2, 4, 4))
        mean_out_complex, cov_out = dft.gaussian_forward(mean=mean_complex, cov=cov)
        fft_out = t.fft.ifft(mean_complex[:, :2] + 1j * mean_complex[:, 2:])
        assert t.allclose(mean_out_complex, t.cat((fft_out.real, fft_out.imag), dim=1))
        assert t.allclose(cov_out, cov)

    def test_gaussian_forward_diag_covariance(self):
        # N=4
        dft = DFT(N=4, precision='float')
        mean_in = t.randn(size=(3,4), dtype=t.cfloat)
        cov_in = t.diag_embed(t.ones(size=(3,8), dtype=t.float))
        mean_out, cov_out = dft.gaussian_forward(mean=t.cat((mean_in.real, mean_in.imag), dim=1), cov=cov_in)
        assert t.allclose(t.fft.fft(mean_in), mean_out[:,:4] + 1j*mean_out[:,4:])
        assert t.allclose(4*t.diag_embed(t.ones(size=(3,8))), cov_out)

    def test_gaussian_backward_diag_covariance(self):
        # N=4
        dft = DFT(N=4, inverse=True, precision='float')
        mean_in = t.randn(size=(3,4), dtype=t.cfloat)
        cov_in = t.diag_embed(t.ones(size=(3,8), dtype=t.float))
        mean_out, cov_out = dft.gaussian_forward(mean=t.cat((mean_in.real, mean_in.imag), dim=1), cov=cov_in)
        assert t.allclose(t.fft.ifft(mean_in), mean_out[:,:4] + 1j*mean_out[:,4:])
        assert t.allclose((1./4)*t.diag_embed(t.ones(size=(3,8))), cov_out)

    def test_gaussian_forward_diag_covariance_interleaved(self):
        # N=4
        dft = DFT(N=4, complex_interleaved=True, precision='float')
        mean_in = t.randn(size=(3,4), dtype=t.cfloat)
        cov_in = t.diag_embed(t.ones(size=(3,8), dtype=t.float))
        mean_out, cov_out = dft.gaussian_forward(mean=t.stack((mean_in.real, mean_in.imag), dim=2).view(3,8), cov=cov_in)
        assert t.allclose(t.fft.fft(mean_in), mean_out[:,::2] + 1j*mean_out[:,1::2])
        assert t.allclose(4*t.diag_embed(t.ones(size=(3,8))), cov_out)

    def test_gaussian_backward_diag_covariance_interleaved(self):
        # N=4
        dft = DFT(N=4, inverse=True, complex_interleaved=True, precision='float')
        mean_in = t.randn(size=(3,4), dtype=t.cfloat)
        cov_in = t.diag_embed(t.ones(size=(3,8), dtype=t.float))
        mean_out, cov_out = dft.gaussian_forward(mean=t.stack((mean_in.real, mean_in.imag), dim=2).view(3,8), cov=cov_in)
        assert t.allclose(t.fft.ifft(mean_in), mean_out[:,::2] + 1j*mean_out[:,1::2])
        assert t.allclose((1./4)*t.diag_embed(t.ones(size=(3,8))), cov_out)

    def test_full_matrix2_block_diag(self):
        # N=3
        dft = DFT(N=3, complex_interleaved=True, precision='float')
        full_matrix = t.randn(size=(4,6,6), dtype=t.cfloat)
        assert t.allclose(dft.full_matrix2_block_diag(full_matrix), full_matrix[:,[0,0,1,1,2,2,3,3,4,4,5,5],[0,1,0,1,2,3,2,3,4,5,4,5]].view(4,3,2,2))

    def test_block_diag2full_matrix(self):
        # N=3
        dft = DFT(N=3, complex_interleaved=True, precision='float')
        diag_elements = t.abs(t.randn(size=(4,3,2), dtype=t.float))
        block_diag = t.diag_embed(diag_elements)
        full_ref = t.diag_embed(diag_elements.view(4,6))
        assert t.allclose(full_ref, dft.block_diag2full_matrix(block_diag))