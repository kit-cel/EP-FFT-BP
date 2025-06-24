# Copyright 2024, 2025 Luca Schmid.
import torch as t
from math import sqrt, pi
from collections import namedtuple

class OfdmSensingChannel:
    """
    OFDM channel with simulation and estimation functionalities.
    """
    def __init__(self, ofdm_size, dev=t.device('cpu')):
        """
        :param ofdm_size: Number of OFDM subcarriers.
        :param dev: Device for tensors.
        """
        self.ofdm_size = ofdm_size
        self.dev = dev
        self.datasample = namedtuple("datasample", "tx_f llrs ber h_t rx_noisefree_f sigma2_noise_sensing_sample")
        self.data = namedtuple("data", "tx_f llrs ber h_t rx_f sigma2_noise_sensing")

    def sample2data(self, datasample, snr_db_sensing):
        """
        :param datasample:
        :param snr_db_sensing: Float in (-inf, +inf) specifying the variance of AWGN at the receiver of the sensing channel in the frequency domain.
        :return:
        """
        sigma2_noise_sensing = t.mean(t.abs(datasample.rx_noisefree_f)**2, dim=-1) / 10**(snr_db_sensing/10) # Variance of AWGN for each batch element
        rx_f = datasample.rx_noisefree_f + t.sqrt(sigma2_noise_sensing).unsqueeze(-1) * datasample.sigma2_noise_sensing_sample  # Add noise

        return self.data(datasample.tx_f, datasample.llrs, datasample.ber, datasample.h_t, rx_f, sigma2_noise_sensing)

    def mmse(self, data, sigma2_h_prior):
        """
        Based on the channel observation data.rx_f, the LLRs of the transmit symbols data.llrs, the variance of the
        sensing noise data.sigma2_noise_semsing and the variance of the assumed Gaussian prior of h sigma2_h_prior,
        this method computes the mean and variance of the channel h_f in the frequency domain given the channel observation.
        The mean is equivalent to the mmse estimator.
        :param data: Batch of simulated transmission.
        :return: Complex-valued mean of shape (batch_size, N) and covariance of shape (batch_size, N, 2, 2).
        """
        batch_size = data.rx_f.shape[0]

        var_mixture_without_prior = data.sigma2_noise_sensing[:,None]/2 + 4*t.abs(data.rx_f)**2 * t.exp(data.llrs) / (t.exp(data.llrs)+1)**2 # var of mixture before appyling prior
        sigma2_Re = (sigma2_h_prior * var_mixture_without_prior)/(sigma2_h_prior + 2*var_mixture_without_prior)
        sigma2_Im = (data.sigma2_noise_sensing * sigma2_h_prior) / (2*(data.sigma2_noise_sensing + sigma2_h_prior))
        mean = data.rx_f * sigma2_h_prior * ((t.exp(data.llrs) - 1) / (t.exp(data.llrs) + 1)) / (sigma2_h_prior + 2*var_mixture_without_prior)

        cov = t.zeros(size=(batch_size, self.ofdm_size, 2, 2), device=self.dev)
        a = t.angle(data.rx_f)
        cos_a = t.cos(a)
        sin_a = t.sin(a)

        cov[...,0,0] = cos_a**2 * sigma2_Re + sin_a**2 * sigma2_Im[:,None]
        cov[...,0,1] = sin_a * cos_a * (sigma2_Re - sigma2_Im[:,None])
        cov[...,1,0] = t.clone(cov[...,0,1])
        cov[...,1,1] = sin_a**2 * sigma2_Re + cos_a**2 * sigma2_Im[:,None]

        return mean, cov

    def mse(self, data, h_t_est):
        return t.mean(t.abs(data.h_t - h_t_est)**2)

    def ep_update_f_white(self, mu_fft, var_fft, data, gamma_f_old=None, lambda_f_old=None, momentum=0.2):
        """
        EP update in the frequency domain, incorporating the LLR and data.rx_f information (Gaussian mixture).
        <White> means that we assume the covariance of the equivalent real-valued system to be a
        (2*ofdm_size) x (2*ofdm_size) diagonal matrix with constant diagonal value var_fft.
        :param mu_fft: Complex mean of current distribution q. Shape=(batch_size,ofdm_size).
        :param var_fft: Variance of each real/imag component of current distribution q. Shape=(batch_size,).
        :param data: Batch of simulated transmission.
        :param gamma_f_old:  Current/old real-valued gamma parameters from the unnormalized Gaussians t_tilde.
                             Shape=(batch_size, ofdm_size, 2). Note: The last dimension is NOT real/imag part, but BPSK/OBPSK dimension (shifted).
        :param lambda_f_old: Current/old real-valued lambda parameters from the unnormalized Gaussians t_tilde.
                             Shape=(batch_size, ofdm_size, 2). Note: The last dimension is NOT real/imag part, but BPSK/OBPSK dimension (shifted).
        :param momentum: Scalar in [0,1] for damped parameter update. moment=1.0 is full update and moment=0.0 is no update.
        :return: complex mean (shape=(batch_size,ofdm_size)) and variance (shape=(batch_size,ofdm_size,2) in complex plane) of updated q distribution
                 and updated gamma and lambda (both have shape=(batch_size,ofdm_size,2) in BPSK/OBPSK domain).
        """
        assert len(mu_fft.shape) == 2
        assert mu_fft.shape[1] == self.ofdm_size
        assert mu_fft.dtype == t.cfloat
        batch_size = mu_fft.shape[0]
        assert var_fft.shape == (batch_size,) # equal variance along each fft vector (law of large numbers), variance is var of real/imag part
        assert not ((var_fft <= 0).any() or t.isnan(var_fft).any() or t.isinf(var_fft).any())
        alpha = data.rx_f.angle()
        rot = t.exp(-1j*alpha) # shape = (batch_size, ofdm_size)

        # 1.) Compute cavity marginal in shifted domain with complex mean mu_cav (shape=(batch_size,ofdm_size)) and dim-wise variance (shape=(batch_size,ofdm_size,2))
        if gamma_f_old == None: # no old gamma/lambda -> cavity distribution = q
            assert lambda_f_old == None
            mu_cav = mu_fft * rot # rotate -phase(rx_f), shape = (batch_size, ofdm_size), dtype=t.cfloat
            var_cav = var_fft[:,None,None].repeat(1,self.ofdm_size,2) # no rotation required since input is circular symmetry, last dim is for bpsk/orthogonal part # shape (batch_size, 1,2)
        else: # gamma_f_old and lambda_f_old are already defined in the shifted domain
            assert not lambda_f_old == None
            assert lambda_f_old.shape == (batch_size, self.ofdm_size, 2)
            assert not ((lambda_f_old < 0).any() or t.isnan(lambda_f_old).any() or t.isinf(lambda_f_old).any())
            assert gamma_f_old.shape == (batch_size, self.ofdm_size)
            assert gamma_f_old.dtype == t.cfloat
            #cavity_update_mask = (var_fft.view(batch_size,1,1) * lambda_f_old) < 1
            var_cav = var_fft.view(batch_size,1,1) / (1 - var_fft.view(batch_size,1,1) * lambda_f_old)
            mu_cav = var_cav * (((mu_fft*rot) / var_fft.view(batch_size,1)).unsqueeze(-1) - gamma_f_old)
            assert not ((var_cav <= 0).any() or t.isinf(var_cav).any() or t.isnan(var_cav))

        # 2.) Compute mean mu_mix and variance var_mix of the Gaussian mixture distribution (cavity * prior_f)
        # 2.1) start with mixture along the BPSK axis
        var_pm = 1 / (1/var_cav[...,0] + 2/data.sigma2_noise_sensing.unsqueeze(-1)) # variance for each mixture component, shape = (batch_size,ofdm_size)
        w_m_log = -t.log(t.exp(data.llrs)+1) - (mu_cav.real + t.abs(data.rx_f)) ** 2 / (2 * (var_cav[...,0] + data.sigma2_noise_sensing.unsqueeze(-1)/2)) # m=minus
        w_p_log = data.llrs - t.log(t.exp(data.llrs)+1) - (mu_cav.real - t.abs(data.rx_f)) ** 2 / (2 * (var_cav[...,0] + data.sigma2_noise_sensing.unsqueeze(-1)/2)) # p=plus

        norm_log = t.maximum(w_m_log, w_p_log) + t.log(1 + t.exp(t.minimum(w_m_log,w_p_log) - t.maximum(w_m_log,w_p_log))) # logSumExp
        a_m = t.exp(w_m_log - norm_log)
        a_p = t.exp(w_p_log - norm_log)
        assert t.allclose(t.ones_like(a_m), a_m + a_p, atol=1e-4)

        mu_bpsk = (var_pm * mu_cav.real)/var_cav[...,0] + t.abs(data.rx_f) * (1-2*a_m)/(1+ data.sigma2_noise_sensing.unsqueeze(-1)/(2*var_cav[...,0]))
        var_bpsk = var_pm + a_m*a_p*((4*var_pm*t.abs(data.rx_f))/data.sigma2_noise_sensing.unsqueeze(-1))**2

        #mu_bpsk = (var_pm * mu_cav.real)/var_cav[...,0] + t.abs(data.rx_f) * t.tanh(data.llrs/2) / (1+ data.sigma2_noise_sensing.unsqueeze(-1)/(2*var_cav[...,0])) # shape (batch_size, ofdm_size), real-valued
        #var_bpsk = var_pm + t.exp(data.llrs)/(t.exp(data.llrs)+1)**2 * ((4*var_pm*t.abs(data.rx_f))/data.sigma2_noise_sensing.unsqueeze(-1))**2 # shape = (batch_size,ofdm_size)

        # 2.2) orthogonal to BPSK axis is a simple multiplication of 2 Gaussians (no mixture)
        var_obpsk = 1 / (1/var_cav[...,1] + 2/data.sigma2_noise_sensing.unsqueeze(-1)) # shape (batch_size, ofdm_size)
        mu_obpsk = mu_cav.imag / (1 + (2*var_cav[...,1])/data.sigma2_noise_sensing.unsqueeze(-1)) # shape (batch_size, ofdm_size), real-valued

        # 3.) Compute parameter updates for gamma and lambda, shape = (batch_size,ofdm_size,2), last dim is bpsk/obpsk dim
        lambda_f_new = 1/t.stack((var_bpsk, var_obpsk), dim=-1) - 1/var_cav
        gamma_f_new = t.stack((mu_bpsk, mu_obpsk), dim=-1)/t.stack((var_bpsk, var_obpsk), dim=-1) - t.view_as_real(mu_cav)/var_cav
        param_update_mask = (lambda_f_new > 0)
        assert not (t.isinf(lambda_f_new).any() or t.isnan(lambda_f_new).any())
        assert not (t.isinf(gamma_f_new).any() or t.isnan(gamma_f_new).any())

        # 4.) Smooth parameter update for the parameters from param_update_mask
        if gamma_f_old == None: # no old params -> full update
            #assert not (~param_update_mask).any()
            gamma_f_updated = gamma_f_new
            lambda_f_updated = lambda_f_new
        else:
            gamma_f_updated = t.clone(gamma_f_old)
            lambda_f_updated = t.clone(lambda_f_old)
            assert (momentum <= 1.0) and (momentum >= 0.0)
            gamma_f_updated[param_update_mask] = momentum * gamma_f_new[param_update_mask] + (1-momentum) * gamma_f_old[param_update_mask]
            lambda_f_updated[param_update_mask] = momentum * lambda_f_new[param_update_mask] + (1-momentum) * lambda_f_old[param_update_mask]
        assert not (t.isinf(lambda_f_updated).any() or t.isnan(lambda_f_updated).any())
        assert not (t.isinf(gamma_f_updated).any() or t.isnan(gamma_f_updated).any())

        # Compute cavity * t_tilde(updated params) [still in rotated domain]
        var_q_updated = 1 / (1/var_cav + lambda_f_updated) # shape=(batch_size,ofdm_size,2)
        mu_q_updated = var_q_updated * (t.view_as_real(mu_cav)/var_cav + gamma_f_updated) # shape=(batch_size,ofdm_size,2)
        assert not ((var_q_updated <= 0).any() or t.isinf(var_q_updated).any() or t.isnan(var_q_updated).any())
        assert not (t.isinf(mu_q_updated).any() or t.isnan(mu_q_updated).any())

        # Rotate mu_q_updated and var_q_updated back.
        mu_q_out = t.view_as_complex(mu_q_updated) * rot.conj() # shape=(batch_size, ofdm_size), complex-valued
        var_q_out = t.stack((
            t.stack((var_q_updated[...,0]*t.cos(alpha)**2 + var_q_updated[...,1]*t.sin(alpha)**2, (var_q_updated[...,0]-var_q_updated[...,1])*t.cos(alpha)*t.sin(alpha)), dim=-1),
            t.stack(((var_q_updated[...,0]-var_q_updated[...,1])*t.cos(alpha)*t.sin(alpha), var_q_updated[...,1]*t.cos(alpha)**2 + var_q_updated[...,0]*t.sin(alpha)**2), dim=-1)
        ), dim=-2) # shape=(batch_size, ofdm_size, 2,2). Last 2 dimensions are complex covariance matrices.

        return mu_q_out, var_q_out, gamma_f_updated, lambda_f_updated

class OfdmSensingChannel_sparse_Gaussian(OfdmSensingChannel):
    """
    Channel as in [1, Sec.V.C].

    [1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
    International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC),
    Surrey, UK, July 2025.
    """
    def __init__(self, ofdm_size, dev=t.device('cpu')):
        super().__init__(ofdm_size=ofdm_size, dev=dev)  # Init parent class and inherit all methods and properties.

    def simulate_transmission_sample(self, sparsity : float, communication_quality : float, batch_size : int, sigma2_no_target : float =0.01, sigma2_target : float =1.0):
        """
        Generate a datasample for simulation purposes.
        The sample can be transformed to a dataset with self.sample2data
        :param sparsity: probability of a tap of the channel impulse response to contain a target (i.e., to have high power)
        :param communication_quality: called c in [1], determines the confidence of the LLR values
        :param batch_size: batch size
        :param sigma2_no_target: power of the 'no target' channel taps
        :param sigma2_target: power of the 'target' channel taps
        :return:
        """
        assert communication_quality > 0
        assert (0 <= sparsity) and (sparsity <= 1)
        assert batch_size > 0
        # Simulation of communication channel
        llrs = (communication_quality * (-1) ** t.randint(low=0, high=2, size=(batch_size, self.ofdm_size), dtype=t.float)) + (sqrt(2 * communication_quality) * t.randn(size=(batch_size, self.ofdm_size))) # LLRs of BPSK symbols at the receiver (e.g., results of soft-output detection and/or decoding of communication channel
        tx_f = t.where(t.rand(size=(batch_size, self.ofdm_size)) < 1/(t.exp(llrs)+1), input=-t.ones(size=(batch_size, self.ofdm_size), dtype=t.cfloat), other=t.ones(size=(batch_size, self.ofdm_size), dtype=t.cfloat))
        ber = t.sum(t.abs(tx_f - t.sign(llrs)))/(2*tx_f.numel()) # ber of the communication channel for hard decision

        # Simulation of sensing channel
        h_t = t.zeros(size=(batch_size, self.ofdm_size), dtype=t.cfloat)
        target_mask = t.rand(size=h_t.shape) > (1. - sparsity)
        h_t[~target_mask] = t.randn_like(h_t[~target_mask]) * sqrt(sigma2_no_target)
        h_t[target_mask] = t.randn_like(h_t[target_mask]) * sqrt(sigma2_target)
        h_f = t.fft.fft(h_t, n=self.ofdm_size)  # Frequency response of channel
        rx_noisefree_f = tx_f * h_f  # Received sensing symbols in frequency domain (noise free)

        sigma2_noise_sensing_sample = t.randn_like(rx_noisefree_f, dtype=t.cfloat)
        return self.datasample(tx_f, llrs, ber, h_t, rx_noisefree_f, sigma2_noise_sensing_sample)
