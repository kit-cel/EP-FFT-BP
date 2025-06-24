# Copyright 2024, 2025 Luca Schmid.
import pandas as pd
import numpy as np
import torch as t
t.manual_seed(0)

from EPFFT import EPFFT, EPFFTindependentGaussian, EPFFTindependentGaussianMixture
from constellation import constellation, bpsk_mapping

ctype = t.cdouble
rtype = t.double

"""
Application of the EP-FFT framework to the fundamental problem of symbol detection 
in a linear inter-symbol interference channel.
This reproduces the results in [1, Sec.V.B].

Restrictions: So far only implemented and tested for BPSK, but code can be easily expanded to higher-order constellations.

[1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC), 
Surrey, UK, July 2025.
"""

def compute_lmmse_kernel(channel, snr_lin, order, n1):
    """Helper function to calculate the kernel of the FIR filter for LMMSE equalization."""
    sigma_w = 1 / snr_lin
    H = t.zeros((order, order + channel.shape[0] - 1), dtype=t.double)
    for i, h in enumerate(H):
        h[i:i + channel.shape[0]] = t.flip(channel, dims=(0,))
    return t.matmul(t.linalg.inv(sigma_w * t.eye(order) + t.matmul(H, H.T)), H[:, -(n1 + 1)])

################################################
# Simulation parameters
################################################
batch_size = 100
N = 1024 # FFT size
block_len = 1000 # actual block length, should be < N, rest is zero-padded
const = constellation(mapping=bpsk_mapping, device=t.device('cpu')) # constellation object
snr_dB_range = t.arange(0, 10.1, 1.0) # SNR points in dB
h = t.tensor([0.04, -0.05, 0.07, -0.21, -0.5, 0.72, 0.36, 0.0, 0.21, 0.03, 0.07]) # channel impulse response h
L = h.shape[-1]-1 # memory
result_path = 'results.csv'
#################################################

# Zero padding and DFT
assert block_len+L+1 <= N
h_t = t.zeros((N,), dtype=ctype)
h_t[:L+1] = h
h_f = t.fft.fft(h_t)
indices_tx = t.randint(low=0, high=const.M, size=(batch_size, block_len))
x_t = t.cat(( const.mapping[indices_tx], t.zeros((batch_size, N-block_len), dtype=ctype)), dim=-1)
x_f = t.fft.fft(x_t)

# Instantiate arrays for results. ser = symbol error rate
ser_dftep = np.zeros(len(snr_dB_range))
ser_fftep_parallel_keepBP = np.zeros(len(snr_dB_range))
ser_zf = np.zeros(len(snr_dB_range))
ser_lmmse_filter = np.zeros(len(snr_dB_range))
ser_lmmse = np.zeros(len(snr_dB_range))

for snr_i, snr_dB in enumerate(snr_dB_range): # loop over SNR points
    print(f"SNR = {snr_dB} dB")
    sigma2_noise_t = 10**(-snr_dB/10)
    noise_t = t.cat((np.sqrt(sigma2_noise_t) * t.randn(size=(batch_size, block_len), dtype=ctype), t.zeros((batch_size, N-block_len), dtype=ctype)), dim=-1)
    noise_f = t.fft.fft(noise_t)
    rx_noiseless_f = x_f * h_f # In the frequency domain, the convolution with h^t is a multiplication.
    rx_f = rx_noiseless_f + noise_f # AWGN
    rx_t = t.fft.ifft(rx_f)
    ##########################
    # EP-DFT:
    ##########################
    zf_vars = (N * sigma2_noise_t / (2 * t.abs(h_f)**2)).view(N,1).repeat(1,2)
    assert not (zf_vars <= 0).any()
    # Define likelihood distribution in frequency domain.
    prior_f = EPFFTindependentGaussian(mu_prior=t.view_as_real(rx_f/h_f),
                                       cov_prior=t.diag_embed(zf_vars).view(1,N,2,2).repeat(batch_size,1,1,1))
    # Define discrete priors in time domain.
    prior_t = EPFFTindependentGaussianMixture(GM_weights=(1/const.M)*t.ones((batch_size,N,const.M),dtype=rtype),
                                              mu_GM_components=t.cat((t.view_as_real(const.mapping).view(1,1,const.M,2).repeat(batch_size, block_len,1,1), t.zeros((batch_size, N-block_len,const.M,2), dtype=rtype)), dim=1),
                                              cov_GM_components=t.diag_embed(1e-6*t.ones(2, dtype=rtype)).view(1,1,1,2,2).repeat(batch_size,N,const.M,1,1))
    dftep = EPFFT(fft_size=N, prior_t=prior_t, prior_f=prior_f)
    # Run EP-DFT with L=4 EP iterations and smoothing=0.5 (beta in [1])
    x_hat_t, _, _, _ = dftep.forward_DFT_parallel(n_ep_iters=4, smoothing=0.5, verbose=False)
    ser_dftep[snr_i] = t.sum(indices_tx[:,:block_len] != const.nearest_neighbor(x_hat_t[:,:block_len])) / (batch_size*block_len)

    ##########################
    # EP-FFT
    ##########################
    zf_vars = (N * sigma2_noise_t / (2 * t.abs(h_f)**2)).view(N,1).repeat(1,2)
    assert not (zf_vars <= 0).any()
    # Define likelihood distribution in frequency domain.
    prior_f = EPFFTindependentGaussian(mu_prior=t.view_as_real(rx_f/h_f),
                                       cov_prior=t.diag_embed(zf_vars).view(1,N,2,2).repeat(batch_size,1,1,1))
    # Define discrete priors in time domain.
    prior_t = EPFFTindependentGaussianMixture(GM_weights=(1/const.M)*t.ones((batch_size,N,const.M),dtype=rtype),
                                              mu_GM_components=t.cat((t.view_as_real(const.mapping).view(1,1,const.M,2).repeat(batch_size, block_len,1,1), t.zeros((batch_size, N-block_len,const.M,2), dtype=rtype)), dim=1),
                                              cov_GM_components=t.diag_embed(1e-6*t.ones(2, dtype=rtype)).view(1,1,1,2,2).repeat(batch_size,N,const.M,1,1))
    fftep = EPFFT(fft_size=N, prior_t=prior_t, prior_f=prior_f, convergence_tol=1e-2)
    x_hat_t, _, _, _ = fftep.forward_FFTBP_parallel(n_ep_iters=4, smoothing=0.5, reset_BP_msgs=False, verbose=True)
    ser_fftep_parallel_keepBP[snr_i] = t.sum(indices_tx[:,:block_len] != const.nearest_neighbor(x_hat_t[:,:block_len])) / (batch_size*block_len)

    ##########################
    # Baseline 1: zero-forcing (ZF) in the frequency domain
    ##########################
    x_hat_f_zf = rx_f/h_f
    x_hat_t_zf_idx = const.nearest_neighbor(t.fft.ifft(x_hat_f_zf))[:,:block_len]
    ser_zf[snr_i] = t.sum(indices_tx[:,:block_len] != x_hat_t_zf_idx) / (batch_size*block_len)

    ##########################
    # Baseline 2: LMMSE equalizer in time domain (FIR filter)
    ##########################
    h_lmmse_kernel = compute_lmmse_kernel(h.real, snr_lin=1/sigma2_noise_t, order=30, n1=(30 - 1) // 2 + 1)
    x_lmmse_eq_t = t.nn.functional.conv1d(rx_t.view(batch_size,1,-1), h_lmmse_kernel.to(ctype).view(1,1,-1), padding=15).view(batch_size,-1)[:,1:]
    x_lmmse_hard_t_idx = const.nearest_neighbor(x_lmmse_eq_t)[:,:block_len]
    ser_lmmse_filter[snr_i] = t.sum(indices_tx[:, :block_len] != x_lmmse_hard_t_idx) / (batch_size * block_len)

    ##########################
    # Baseline 3: LMMSE equalizer in frequency domain (EP with 0 iterations)
    ##########################
    zf_vars = (N * sigma2_noise_t / (2 * t.abs(h_f)**2)).view(N,1).repeat(1,2)
    assert not (zf_vars <= 0).any()
    prior_f = EPFFTindependentGaussian(mu_prior=t.view_as_real(rx_f/h_f),
                                       cov_prior=t.diag_embed(zf_vars).view(1,N,2,2).repeat(batch_size,1,1,1))
    prior_t = EPFFTindependentGaussianMixture(GM_weights=(1/const.M)*t.ones((batch_size,N,const.M),dtype=rtype),
                                              mu_GM_components=t.cat((t.view_as_real(const.mapping).view(1,1,const.M,2).repeat(batch_size, block_len,1,1), t.zeros((batch_size, N-block_len,const.M,2), dtype=rtype)), dim=1),
                                              cov_GM_components=t.diag_embed(1e-6*t.ones(2, dtype=rtype)).view(1,1,1,2,2).repeat(batch_size,N,const.M,1,1))
    dftep = EPFFT(fft_size=N, prior_t=prior_t, prior_f=prior_f)
    x_hat_t, _, _, _ = dftep.forward_DFT_parallel(n_ep_iters=1, smoothing=0.0, verbose=False)
    ser_lmmse[snr_i] = t.sum(indices_tx[:,:block_len] != const.nearest_neighbor(x_hat_t[:,:block_len])) / (batch_size*block_len)

    print(f"EP-DFT={ser_dftep[snr_i]}, EP-FFT={ser_fftep_parallel_keepBP[snr_i]}, LMMSE={ser_lmmse[snr_i]}, LMMSE filter={ser_lmmse_filter[snr_i]}")

    # Write results to csv file.
    results = {f"SNR (dB)" : snr_dB_range,
               f"ZF" : ser_zf,
               f"LMMSE FIR" : ser_lmmse_filter,
               f"LMMSE" : ser_lmmse,
               f"DFTEP" : ser_dftep,
               f"EPFFT" : ser_fftep_parallel_keepBP,}

    pd.DataFrame(results).to_csv(result_path)
