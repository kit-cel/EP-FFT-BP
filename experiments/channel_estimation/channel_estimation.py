# Copyright 2024, 2025 Luca Schmid.
import numpy as np
import pandas as pd
import torch as t
dev = t.device('cpu')
t.manual_seed(0)

from EPFFT import EPFFT, EPFFTindependentGaussianMixture
from Channels import OfdmSensingChannel_sparse_Gaussian
from Utils import llrs2hardZF_f

"""
Application of the EP-FFT framework to the problem of channel estimation in a channel with multiple dominant reflectors 
leading to a sparse power-delay profile, e.g., in the context of a multistatic joint communication and sensing scenario.
This reproduces the results in [1, Sec.V.C].

Restrictions: So far only implemented and tested for BPSK, but code can be easily expanded to higher-order constellations.

[1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC), 
Surrey, UK, July 2025.
"""

################################################
# Simulation parameters
################################################
ofdm_size = 1024 # OFDM and FFT size
sparsity = 0.01 # probability of a tap of the channel impulse response to contain a target (i.e., to have high power)
batch_size = 100 # batch size
communication_quality_range = [3.25,] # list of parameterers (called c in the paper) for the LLR sampling to simulate
snr_db_sensing_range = np.arange(-20., 20.1, step=2.0)
ep_iters_range = [4,] # list of # EP iters to simulate (L in the paper)
fftbp_convergence_tol_range = np.array([1e-2,])
result_path = f"results.csv" # path to save results
#################################################
sens_channel = OfdmSensingChannel_sparse_Gaussian(ofdm_size=ofdm_size)
results = {f"Sensing SNR (dB)" : snr_db_sensing_range}

# Instantiate arrays for results.
mse_zf_hard = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range))) # hard decision + zero forcing in freq domain, then IFFT
mse_mmse_GaussAssmptionHf = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range))) # mmse in freq domain with gaussian assumption on h_f, then IFFT of mean
mse_mmse_NoAssmptionHf = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range))) # mmse in freq domain with gaussian assumption on h_f, then IFFT of mean
mse_dftep = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range), len(ep_iters_range)))
mse_fftep_layeredBP = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range), len(ep_iters_range), len(fftbp_convergence_tol_range)))
mse_fftep_floodingBP = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range), len(ep_iters_range), len(fftbp_convergence_tol_range)))
ber_fftep_floodingBP = np.zeros(shape=(len(snr_db_sensing_range), len(communication_quality_range), len(ep_iters_range), len(fftbp_convergence_tol_range)))

for comm_quali_i, communication_quality in enumerate(communication_quality_range):
    datasample = sens_channel.simulate_transmission_sample(sparsity=sparsity, communication_quality=communication_quality, batch_size=batch_size)
    print(f"Simulating communication quality {communication_quality} with BER={datasample.ber}.")
    for snr_i, snr_db_sensing in enumerate(snr_db_sensing_range):
        print(f"  SNR = {snr_db_sensing} dB")
        data = sens_channel.sample2data(datasample=datasample, snr_db_sensing=snr_db_sensing)

        ################################################################################################################
        # Baseline 1: ZF estimator based on hard decision, then IFFT
        ################################################################################################################
        chanest_zf_hard_f = llrs2hardZF_f(llrs=data.llrs, rx_f=data.rx_f)
        chanest_zf_hard_t = t.fft.ifft(chanest_zf_hard_f)
        mse_zf_hard[snr_i, comm_quali_i] = sens_channel.mse(data=data, h_t_est = chanest_zf_hard_t)
        
        ################################################################################################################
        # Baseline 2: MMSE estimator (no prior assumption), then hard decision in freq domain and IFFT
        ################################################################################################################
        chanest_mmse_NoAssmptionHf, _ = sens_channel.mmse(data=data, sigma2_h_prior=1e8)#5*sparsity*ofdm_size)
        mse_mmse_NoAssmptionHf[snr_i, comm_quali_i] = sens_channel.mse(data=data, h_t_est=t.fft.ifft(chanest_mmse_NoAssmptionHf))

        ################################################################################################################
        # Baseline 3: MMSE estimator (Gaussian prior assumption), then hard decision in freq domain and IFFT
        ################################################################################################################
        chanest_mmse_GaussAssmptionHf, _ = sens_channel.mmse(data=data, sigma2_h_prior=ofdm_size * (sparsity + (1-sparsity)*0.01))
        mse_mmse_GaussAssmptionHf[snr_i, comm_quali_i] = sens_channel.mse(data=data, h_t_est=t.fft.ifft(chanest_mmse_GaussAssmptionHf))

        # save intermediate results..
        results.update({f"c{comm_quali_i} ZF" : mse_zf_hard[:,comm_quali_i],
                    f"c{comm_quali_i} MMSE GaussAssumptionHf" : mse_mmse_GaussAssmptionHf[:,comm_quali_i],
                    f"c{comm_quali_i} MMSE noAssumptionHf" : mse_mmse_NoAssmptionHf[:,comm_quali_i],
                    })
        pd.DataFrame(results).to_csv(result_path)

        for ep_iters_i, ep_iters in enumerate(ep_iters_range):
            ################################################################################################################
            # EP-DFT
            ################################################################################################################
            w_minus = (1/(t.exp(data.llrs)+1)).double()
            prior_f = EPFFTindependentGaussianMixture(GM_weights=t.stack([w_minus, 1-w_minus], dim=-1),
                                                      mu_GM_components=t.view_as_real(t.stack([-data.rx_f, data.rx_f], dim=-1)).double(),
                                                      cov_GM_components=(t.diag_embed(data.sigma2_noise_sensing.unsqueeze(-1)/2 * t.ones((1,2), dtype=t.double))).view(batch_size,1,1,2,2).repeat(1,ofdm_size,2,1,1))
            prior_t = EPFFTindependentGaussianMixture(GM_weights=t.tensor([sparsity, 1-sparsity]).view(1,1,2).repeat(batch_size,ofdm_size,1).double(),
                                                      mu_GM_components=t.zeros((batch_size,ofdm_size,2,2), dtype=t.double),
                                                      cov_GM_components=t.stack([t.diag_embed(0.5*t.ones((batch_size,ofdm_size,2), dtype=t.double)), t.diag_embed(0.005*t.ones((batch_size,ofdm_size,2), dtype=t.double))], dim=2))
            dftep = EPFFT(fft_size=ofdm_size, prior_t=prior_t, prior_f=prior_f)
            chanest_dftep_t, _, _, _ = dftep.forward_DFT_parallel(n_ep_iters=ep_iters, verbose=False, smoothing=0.5)
            mse_dftep[snr_i, comm_quali_i, ep_iters_i] = sens_channel.mse(data=data, h_t_est=chanest_dftep_t)

            # save intermediate results..
            results.update({f"c{comm_quali_i} DFTEP {ep_iters}": mse_dftep[:, comm_quali_i, ep_iters_i]})
            pd.DataFrame(results).to_csv(result_path)

            ################################################################################################################
            # EP-FFT
            ################################################################################################################
            for fftbp_convergence_tol_i, fftbp_convergence_tol in enumerate(fftbp_convergence_tol_range):
                # layered GaBP schedule (not shown in paper [1])
                print(f"    FFTBP layered")
                w_minus = (1/(t.exp(data.llrs)+1)).double()
                prior_f = EPFFTindependentGaussianMixture(GM_weights=t.stack([w_minus, 1-w_minus], dim=-1),
                                                          mu_GM_components=t.view_as_real(t.stack([-data.rx_f, data.rx_f], dim=-1)).double(),
                                                          cov_GM_components=(t.diag_embed(data.sigma2_noise_sensing.unsqueeze(-1)/2 * t.ones((1,2), dtype=t.double))).view(batch_size,1,1,2,2).repeat(1,ofdm_size,2,1,1))
                prior_t = EPFFTindependentGaussianMixture(GM_weights=t.tensor([sparsity, 1-sparsity]).view(1,1,2).repeat(batch_size,ofdm_size,1).double(),
                                                          mu_GM_components=t.zeros((batch_size,ofdm_size,2,2), dtype=t.double),
                                                          cov_GM_components=t.stack([t.diag_embed(0.5*t.ones((batch_size,ofdm_size,2), dtype=t.double)), t.diag_embed(0.005*t.ones((batch_size,ofdm_size,2), dtype=t.double))], dim=2))
                fftep = EPFFT(fft_size=ofdm_size, prior_t=prior_t, prior_f=prior_f, convergence_tol=fftbp_convergence_tol)
                chanest_fftep_t, _, _, _ = fftep.forward_FFTBP_parallel(n_ep_iters=ep_iters, smoothing=0.5, BP_schedule='layered', reset_BP_msgs=False, verbose=True)
                mse_fftep_layeredBP[snr_i, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i] = sens_channel.mse(data=data, h_t_est=chanest_fftep_t)

                # flooding schedule -> results in Fig. 5 of paper [1]
                print(f"    FFTBP flooding")
                w_minus = (1/(t.exp(data.llrs)+1)).double()
                prior_f = EPFFTindependentGaussianMixture(GM_weights=t.stack([w_minus, 1-w_minus], dim=-1),
                                                          mu_GM_components=t.view_as_real(t.stack([-data.rx_f, data.rx_f], dim=-1)).double(),
                                                          cov_GM_components=(t.diag_embed(data.sigma2_noise_sensing.unsqueeze(-1)/2 * t.ones((1,2), dtype=t.double))).view(batch_size,1,1,2,2).repeat(1,ofdm_size,2,1,1))
                prior_t = EPFFTindependentGaussianMixture(GM_weights=t.tensor([sparsity, 1-sparsity]).view(1,1,2).repeat(batch_size,ofdm_size,1).double(),
                                                          mu_GM_components=t.zeros((batch_size,ofdm_size,2,2), dtype=t.double),
                                                          cov_GM_components=t.stack([t.diag_embed(0.5*t.ones((batch_size,ofdm_size,2), dtype=t.double)), t.diag_embed(0.005*t.ones((batch_size,ofdm_size,2), dtype=t.double))], dim=2))
                fftep = EPFFT(fft_size=ofdm_size, prior_t=prior_t, prior_f=prior_f, convergence_tol=fftbp_convergence_tol)
                chanest_fftep_t, _, symbolDet_fft_f, _ = fftep.forward_FFTBP_parallel(n_ep_iters=ep_iters, smoothing=0.5, BP_schedule='flooding', reset_BP_msgs=False, verbose=True)
                mse_fftep_floodingBP[snr_i, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i] = sens_channel.mse(data=data, h_t_est=chanest_fftep_t)
                ber_fftep_floodingBP[snr_i, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i] = t.sum(t.abs(data.tx_f - t.sign((symbolDet_fft_f/t.fft.fft(chanest_fftep_t)).real)))/(2*data.tx_f.numel())

                # save intermediate results..
                results.update({f"c{comm_quali_i} EPFFT {ep_iters} BPlayered ctol {fftbp_convergence_tol}" : mse_fftep_layeredBP[:, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i],
                                f"c{comm_quali_i} EPFFT {ep_iters} BPflooding ctol {fftbp_convergence_tol}" : mse_fftep_floodingBP[:, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i],
                                f"BER c{comm_quali_i} EPFFT {ep_iters} BPflooding ctol {fftbp_convergence_tol}" : ber_fftep_floodingBP[:, comm_quali_i, ep_iters_i, fftbp_convergence_tol_i]
                            })
                pd.DataFrame(results).to_csv(result_path)


pd.DataFrame(results).to_csv(result_path)
print(f"Finished and saved.")
