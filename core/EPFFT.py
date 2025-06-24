# Copyright 2024, 2025 Luca Schmid.
import torch as t
from core.DFT import DFT
from core.FFTBP import FFTBP

class EPFFT:
    """
    Expectation Propagation (EP) framework to find a good Gaussian approximation of a probabilistic system involving
    the DFT/FFT.
    This code is part of the EP-FFT framework [1].

    [1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
    International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC),
    Surrey, UK, July 2025.
    """
    def __init__(self, fft_size: int, prior_t, prior_f, dev: t.device = t.device('cpu'), prec: str = 'double', convergence_tol=1e-5):
        """
        Initializes an instance of the EPFFT class with specified FFT size.
        :param int fft_size: Size of the FFT.
        :param t.device dev: Device on which torch.tensor are stored, e.g., t.device('cpu') or t.device('cuda').
        :param str prec: Either single precision 'float' or double precision 'double'.
        """
        if not isinstance(fft_size, int): raise ValueError(f'Input argument fft_size must be integer but has type {type(fft_size)}.')
        self.fft_size = fft_size
        assert prior_t.fft_size == self.fft_size and prior_f.fft_size == self.fft_size
        self.prior_t = prior_t
        self.prior_f = prior_f
        assert prior_t.bs == prior_f.bs
        self.bs = prior_t.bs

        # Set up DFT object for the EP-DFT algorithm.
        self.dft =  DFT(N=fft_size, inverse=False, complex_interleaved=True, dev=dev, precision=prec)
        self.idft = DFT(N=fft_size, inverse=True, complex_interleaved=True, dev=dev, precision=prec)

        # Set up FFT/GaBP object for the EP-FFT algorithm.
        self.fft = FFTBP(N=fft_size, dev=dev, prec=prec, convergence_tol=convergence_tol)

        self.dev = dev
        if prec == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif prec == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {prec}.")

    def forward_FFTBP_pingPong(self, start_in_time_domain: bool, n_ep_iters: int, smoothing=0.0, verbose: bool=False):
        """
        Runs the EP algorithm with specified prior distributions in time/frequency domain
        using the EP-FFT framework (i.e., Gaussian BP in the FFT) to switch domains (N log N complexity).
        This function uses a ping pong schedule for EP, i.e., the EP parameters of time and frequency domain
        are updated alternately.
        :param bool start_in_time_domain: If True, the EP iterations start in the time domain, else in the frequency domain.
        :param n_ep_iters: Number of iterations. One iteration means propagating the uncertainty forward and backward
        throught the DFT/IDFT.
        :param smoothing: Scalar in [0,1] to smooth the EP parameter update of gamma/lambda based on the old parameters.
        :param verbose:  If true, prints some statements, like convergence info.
        :return: res_mu_t, res_cov_t, res_mu_f, res_cov_f: Mean and covariance for each variable
                 in time and frequency domain, respectively.
        """
        assert n_ep_iters >= 0

        if start_in_time_domain: # Incorporate priors in the time domain.
            self._ep_update(self.prior_t,
                            mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                            cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)
            n_iters = n_ep_iters

        else: # Start with initial zero-information signal in time domain which is transformed to frequency domain.
            self.prior_t.lam = t.diag_embed(1e-12*t.ones(size=(self.bs, self.fft_size, 2), dtype=self.prec))
            n_iters = n_ep_iters+1

        for ep_iter_i in range(n_iters):
            # Use FFT-BP to do marginal inference of global Gaussian approximation.
            mu_q_t, cov_q_t, mu_q_f, cov_q_f = self.fft.BP_layered(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                   prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                   max_iters=10*self.fft_size, verbose=True)
            # Update parameters in frequency domain
            self._ep_update(self.prior_f, mu_in=mu_q_f, cov_in=cov_q_f, smoothing=smoothing, verbose=verbose)

            # Use FFT-BP to do marginal inference of global Gaussian approximation.
            mu_q_t, cov_q_t, mu_q_f, cov_q_f = self.fft.BP_layered(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                   prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                   max_iters=10*self.fft_size, verbose=True)
            # Update parameters in time domain
            self._ep_update(self.prior_t, mu_in=mu_q_t, cov_in=cov_q_t, smoothing=smoothing, verbose=verbose)

        return (t.view_as_complex(mu_q_t), cov_q_t, t.view_as_complex(mu_q_f), cov_q_f) # TODO return full 2x2 covs? (DFT version only outputs variance of real/imag part)

    def forward_FFTBP_parallel(self, n_ep_iters: int, smoothing=0.0, BP_schedule='flooding', reset_BP_msgs=True, verbose: bool=False):
        """
        Runs the EP algorithm with specified prior distributions in time/frequency domain
        using the EP-FFT framework (i.e., Gaussian BP in the FFT) to switch domains (N log N complexity).
        This function uses a parallel schedule for EP, i.e., the EP parameters of time and frequency domain
        are always updated in parallel.
        :param n_ep_iters: Number of iterations. One iteration means propagating the uncertainty forward and backward
        throught the DFT/IDFT.
        :param smoothing: Scalar in [0,1] to smooth the EP parameter update of gamma/lambda based on the old parameters.
        :param verbose:  If true, prints some statements, like convergence info.
        :return: res_mu_t, res_cov_t, res_mu_f, res_cov_f: Mean and covariance for each variable
                 in time and frequency domain, respectively.
        """
        assert n_ep_iters >= 0
        assert BP_schedule == 'flooding' or BP_schedule == 'layered'

        self._ep_update(self.prior_t,
                        mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                        cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)

        self._ep_update(self.prior_f,
                        mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                        cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)


        for ep_iter_i in range(n_ep_iters+1):
            # Use FFT-BP to do marginal inference of global Gaussian approximation.
            if BP_schedule == 'layered':
                if reset_BP_msgs:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f = self.fft.BP_layered(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose)
                elif ep_iter_i == 0:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f, current_BP_msgs = self.fft.BP_layered(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose, return_messages=True)
                else:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f, current_BP_msgs = self.fft.BP_layered(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose, init_messages=current_BP_msgs, return_messages=True)
            else:
                if reset_BP_msgs:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f = self.fft.BP_flooding(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose)
                elif ep_iter_i == 0:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f, current_BP_msgs = self.fft.BP_flooding(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose, return_messages=True)
                else:
                    mu_q_t, cov_q_t, mu_q_f, cov_q_f, current_BP_msgs = self.fft.BP_flooding(prior_mu_t=t.linalg.solve(self.prior_t.lam, self.prior_t.gam), prior_cov_t=t.inverse(self.prior_t.lam),
                                                                           prior_mu_f=t.linalg.solve(self.prior_f.lam, self.prior_f.gam), prior_cov_f=t.inverse(self.prior_f.lam),
                                                                           max_iters=10*self.fft_size, verbose=verbose, init_messages=current_BP_msgs, return_messages=True)
            # Update parameters in time and frequency domain in parallel
            self._ep_update(self.prior_t, mu_in=mu_q_t, cov_in=cov_q_t, smoothing=smoothing, verbose=verbose)
            self._ep_update(self.prior_f, mu_in=mu_q_f, cov_in=cov_q_f, smoothing=smoothing, verbose=verbose)


        return (t.view_as_complex(mu_q_t), cov_q_t, t.view_as_complex(mu_q_f), cov_q_f) # TODO return full 2x2 covs? (DFT version only outputs variance of real/imag part)

    def forward_DFT_pingPong(self, start_in_time_domain: bool, n_ep_iters: int, smoothing=0.0, verbose: bool=False):
        """
        Runs the EP algorithm with specified prior distributions in time/frequency domain
        using the DFT matrix to switch domains (cubic complexity).
        :param bool start_in_time_domain: If True, the EP iterations start in the time domain, else in the frequency domain.
        :param n_ep_iters: Number of iterations. One iteration means propagating the uncertainty forward and backward
        throught the DFT/IDFT.
        :param smoothing: Scalar in [0,1] to smooth the EP parameter update of gamma/lambda based on the old parameters.
        :param verbose:  If true, prints some statements, like convergence info.
        :return: res_mu_t, res_cov_t, res_mu_f, res_cov_f: Mean and covariance for each variable
                 in time and frequency domain, respectively.
        """
        assert n_ep_iters >= 0

        if start_in_time_domain: # Incorporate priors in the time domain.
            self._ep_update(self.prior_t,
                            mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                            cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)
            n_iters = n_ep_iters

        else: # Start with initial zero-information signal in time domain which is transformed to frequency domain.
            self.prior_t.lam = t.diag_embed(1e-12*t.ones(size=(self.bs, self.fft_size, 2), dtype=self.prec))
            n_iters = n_ep_iters+1

        # The current global approximation q(x) only consists of the Gaussian approximation of the time-domain part,
        # because the frequency-domain part is not yet incorporated (and initialized with zero precision).
        cov_q_t = self.dft.block_diag2full_matrix(t.inverse(self.prior_t.lam))
        mu_q_t = t.linalg.solve(self.prior_t.lam, self.prior_t.gam).reshape(self.bs, 2 * self.fft_size)

        for ep_iter_i in range(n_iters):
            # Use DFT to transform Gaussian distribution to frequency domain.
            mu_dft, cov_dft = self.dft.gaussian_forward(mean=mu_q_t.view(self.bs, 2 * self.fft_size), cov=cov_q_t)
            # Incorporate priors in the frequency domain.
            lam_f_old = t.clone(self.prior_f.lam)
            gam_f_old = t.clone(self.prior_f.gam)
            self._ep_update(self.prior_f, mu_in=mu_dft.view(self.bs, self.fft_size, 2),
                            cov_in=self.dft.full_matrix2_block_diag(cov_dft), smoothing=smoothing, verbose=verbose)
            # Apply updated local Gaussian approximation q_f (with updated lambda_t/gamma_t) to global system (multipy Gaussian distributions).
            prec_q_f = t.inverse(cov_dft) + self.dft.block_diag2full_matrix(self.prior_f.lam-lam_f_old)
            info_q_f = t.linalg.solve(A=cov_dft, B=mu_dft) + (self.prior_f.gam-gam_f_old).view(self.bs, 2 * self.fft_size)
            cov_q_f = t.inverse(prec_q_f)
            mu_q_f = t.linalg.solve(prec_q_f, info_q_f)

            # Use IDFT to transform global Gaussian approximation to the time domain.
            mu_idft, cov_idft = self.idft.gaussian_forward(mean=mu_q_f, cov=cov_q_f)
            # Incorporate priors in the time domain.
            lam_t_old = t.clone(self.prior_t.lam)
            gam_t_old = t.clone(self.prior_t.gam)
            self._ep_update(self.prior_t, mu_in=mu_idft.view(self.bs, self.fft_size, 2),
                            cov_in=self.idft.full_matrix2_block_diag(cov_idft), smoothing=smoothing, verbose=verbose)
            # Apply updated local Gaussian approximation q_t (with updated lambda_t/gamma_t) to global system (multipy Gaussian distributions).
            prec_q_t = t.inverse(cov_idft) + self.idft.block_diag2full_matrix(self.prior_t.lam-lam_t_old)
            info_q_t = t.linalg.solve(A=cov_idft, B=mu_idft) + (self.prior_t.gam-gam_t_old).view(self.bs, 2 * self.fft_size)
            cov_q_t = t.inverse(prec_q_t)
            mu_q_t = t.linalg.solve(prec_q_t, info_q_t)

        return (mu_q_t[:,::2] + 1j*mu_q_t[:,1::2], t.diagonal(cov_q_t, dim1=-2,dim2=-1).view(self.bs, self.fft_size,2),
                mu_q_f[:,::2] + 1j*mu_q_f[:,1::2], t.diagonal(cov_q_f, dim1=-2,dim2=-1).view(self.bs, self.fft_size,2))

    def forward_DFT_parallel(self, n_ep_iters: int, smoothing=0.0, verbose: bool=False):
        """
        Runs the EP algorithm with specified prior distributions in time/frequency domain
        using the DFT matrix to switch domains (cubic complexity).
        :param bool start_in_time_domain: If True, the EP iterations start in the time domain, else in the frequency domain.
        :param n_ep_iters: Number of iterations. One iteration means propagating the uncertainty forward and backward
        throught the DFT/IDFT.
        :param smoothing: Scalar in [0,1] to smooth the EP parameter update of gamma/lambda based on the old parameters.
        :param verbose:  If true, prints some statements, like convergence info.
        :return: res_mu_t, res_cov_t, res_mu_f, res_cov_f: Mean and covariance for each variable
                 in time and frequency domain, respectively.
        """
        assert n_ep_iters >= 0

        self._ep_update(self.prior_t,
                        mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                        cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)

        self._ep_update(self.prior_f,
                        mu_in=t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec),
                        cov_in=t.diag_embed(1e12 * t.ones(self.bs, self.fft_size, 2, dtype=self.prec)), smoothing=smoothing, verbose=verbose)

        for ep_iter_i in range(n_ep_iters):
            # From updated parameters, build up the new global Gaussian approximation in the time domain.
            mu_f2t, cov_f2t = self.idft.gaussian_forward(mean=t.linalg.solve(self.prior_f.lam, self.prior_f.gam).reshape(self.bs, 2*self.fft_size), cov=self.idft.block_diag2full_matrix(t.inverse(self.prior_f.lam)))
            prec_q_t = t.inverse(cov_f2t) + self.dft.block_diag2full_matrix(self.prior_t.lam)
            info_q_t = t.linalg.solve(cov_f2t, mu_f2t) + self.prior_t.gam.view(self.bs, 2*self.fft_size)
            cov_q_t = t.inverse(prec_q_t)
            mu_q_t = t.linalg.solve(prec_q_t, info_q_t)
            # Use DFT to transform global Gaussian approximation to frequency domain.
            mu_q_f, cov_q_f = self.dft.gaussian_forward(mean=mu_q_t, cov=cov_q_t)

            # EP update: Incorporate priors in both domains and update EP parameters.
            self._ep_update(self.prior_t, mu_in=mu_q_t.view(self.bs, self.fft_size, 2),
                            cov_in=self.dft.full_matrix2_block_diag(cov_q_t), smoothing=smoothing, verbose=verbose)
            self._ep_update(self.prior_f, mu_in=mu_q_f.view(self.bs, self.fft_size, 2),
                            cov_in=self.dft.full_matrix2_block_diag(cov_q_f), smoothing=smoothing, verbose=verbose)

        return (mu_q_t[:,::2] + 1j*mu_q_t[:,1::2], t.diagonal(cov_q_t, dim1=-2,dim2=-1).view(self.bs, self.fft_size,2),
                mu_q_f[:,::2] + 1j*mu_q_f[:,1::2], t.diagonal(cov_q_f, dim1=-2,dim2=-1).view(self.bs, self.fft_size,2))

    def _ep_update(self, prior, mu_in: t.tensor, cov_in: t.tensor, smoothing: float=0.0, verbose=False):
        """
        Given independent 2D Gaussians N(mu_in, cov_in), apply one iteration of the EP algorithm to incorporate
        prior knowledge to the Gaussian approximation.
        If no info_old/prec_old is given (e.g., in the initial EP iteration) and momentum==1.0,
        this returns the (complex) marginal mean and covariance of the multiplication of the given
        independent complex Gaussians N(mu,cov) with the prior distribution.

        Args:
        :param t.tensor mu_in:      Mean of current distribution q.
                                    Real-valued tensor of shape=(batch_size,fft_size,2).
        :param t.tensor cov_in:     2x2 covariances of current distribution q.
                                    Real-valued tensor of shape=(batch_size,fft_size,2,2).
        :param float    smoothing:  Scalar in [0,1] to smooth/dampen the update of the parameters gamma/lambda based on the old parameters:
                                    gamma_updated = (1-smoothing) * gamma_new + smoothing * gamma_old (and same for lambda),
                                    i.e., smoothing=0.0 is a full parameter update and smoothing=1.0 is no update at all.
        """
        assert mu_in.shape == (self.bs, self.fft_size, 2)
        assert mu_in.dtype == self.prec
        assert cov_in.shape == (self.bs, self.fft_size, 2, 2) # equal variance along each fft vector (law of large numbers)

        # 1.) Compute cavity marginal with real/imag mean (shape=(batch_size,fft_size,2)) and dim-wise variance (shape=(batch_size,fft_size,2))
        prec_cav = t.inverse(cov_in) - prior.lam
        info_cav = t.linalg.solve(cov_in, mu_in) - prior.gam
        assert not (t.isinf(prec_cav).any() or t.isnan(prec_cav).any())
        assert not (t.isinf(info_cav).any() or t.isnan(info_cav).any())

        # 2.) Incorporate prior distribution (cavity * prior) and compute mean mu_inc and covariance cov_inc of the resulting distribution.
        info_inc, prec_inc = prior.incorporate(info_cav=info_cav, prec_cav=prec_cav)
        assert prec_inc.shape == (self.bs, self.fft_size, 2, 2) and not ((t.isinf(prec_inc).any() or t.isnan(prec_inc).any()))
        assert info_inc.shape == (self.bs, self.fft_size, 2) and not ((t.isinf(info_inc).any() or t.isnan(info_inc).any()))

        # 3.) Compute parameter updates for gamma and lambda
        lambda_t_new = prec_inc - prec_cav
        gamma_t_new = info_inc - info_cav
        param_update_mask = self._is_psd(lambda_t_new)

        # 4.) Smooth parameter update for the parameters from param_update_mask
        gamma_t_updated = t.clone(prior.gam)
        lambda_t_updated = t.clone(prior.lam)
        assert (smoothing <= 1.0) and (smoothing >= 0.0)
        gamma_t_updated[param_update_mask] = (1-smoothing) * gamma_t_new[param_update_mask] + smoothing * prior.gam[param_update_mask]
        lambda_t_updated[param_update_mask] = (1-smoothing) * lambda_t_new[param_update_mask] + smoothing * prior.lam[param_update_mask]
        assert not (t.isinf(lambda_t_updated).any() or t.isnan(lambda_t_updated).any())
        assert not (t.isinf(gamma_t_updated).any() or t.isnan(gamma_t_updated).any())
        # Write updated parameters to prior object.
        prior.gam = t.clone(gamma_t_updated)
        prior.lam = t.clone(lambda_t_updated)

    def _is_psd(self, mat2x2):
        """
        Given 2x2 matrices, check if all eigenvalues are non-negative.
        :param mat2x2: Tensor of shape (...,2,2).
        :return: True if all eigenvalues of all given matrices are non-negative, False else.
        """
        return t.all(t.linalg.eigvals(mat2x2).real >= 0, dim=-1)

class EPFFTindependentCircularGaussian:
    """
    Class defining a fully factorized circular Gaussian distribution.
    This can be used as input argument (prior_t or prior_f) to the EPFFT class.
    """
    def __init__(self, mu_prior, var_prior, dev: t.device = t.device('cpu'), precision='double'):
        """
        :param t.tensor mu_prior: Mean of the prior distribution. Shape=(batch_size,fft_size,2). Last dim is real/imag part.
        :param t.tensor var_prior: Variance of circular Gaussian distribution in each dimension. Shape=(batch_size,fft_size).
                                   Note that the variance for the real/imag part is var_prior/2, respectively.
        """
        assert len(mu_prior.shape) == 3
        self.bs = mu_prior.shape[0]
        self.fft_size = mu_prior.shape[1]
        assert mu_prior.shape[2] == 2
        assert var_prior.shape == (self.bs, self.fft_size)
        assert not (var_prior <= 0).any()
        self.mu_prior = mu_prior
        self.var_prior = var_prior

        self.dev = dev
        if precision == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif precision == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {precision}.")

        # Initialize lambda and gamma (canonical Gaussian parameters, also called information/precision).
        self.lam = t.diag_embed(1e-24*t.ones(size=(self.bs, self.fft_size, 2), dtype=self.prec))
        self.gam = t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec)

    def incorporate(self, info_cav, prec_cav):
        """
        Given complex Gaussian (cavity) distributions with mean mu_cav and 2x2 covariance matrix cov_cav,
        and independent circular Gaussian priors with mean mu_prior and complex variance var_prior,
        compute the moments of the resulting Gaussian as the product of both input distributions.
        :param t.tensor mu_cav: Mean of the cavity distribution. Shape=(batch_size,fft_size,2). Last dim is real/imag part. # TODO change to info/prec
        :param t.tensor cov_cav: Covariance matrices of the cavity distribution in each (complex) dimension. Shape=(batch_size,fft_size,2,2). # TODO change to info/prec
        :return:
            - mu_out    - Mean of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2).
            - cov_out   - Covariance of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2,2).
        """
        assert info_cav.shape == (self.bs, self.fft_size, 2) and info_cav.dtype == self.prec
        assert prec_cav.shape == (self.bs, self.fft_size, 2, 2)

        # Multiplication of 2 complex Gaussians
        prec_out = prec_cav + t.diag_embed(2/self.var_prior.unsqueeze(-1).repeat(1,1,2))
        #cov_out = t.inverse(prec_out)
        info_out = info_cav + self.mu_prior*2/self.var_prior.unsqueeze(-1)
        #mu_out = t.bmm(prec_out.view(-1,2,2), (info_out).reshape(-1,2,1)).view(self.batch_size, self.fft_size, 2)
        assert not (t.isinf(prec_out).any() or t.isnan(prec_out).any())
        assert not (t.isinf(info_out).any() or t.isnan(info_out).any())

        return info_out, prec_out

class EPFFTindependentGaussian:
    """
    Class defining a fully factorized Gaussian distribution.
    This can be used as input argument (prior_t or prior_f) to the EPFFT class.
    """
    def __init__(self, mu_prior, cov_prior, dev: t.device = t.device('cpu'), precision='double'):
        """
        If the distribution is a
        :param t.tensor mu_prior: Mean of the prior distribution. Shape=(batch_size,fft_size,2). Last dim is real/imag part.
        :param t.tensor cov_prior: Covariance of Gaussian distribution in each dimension. Shape=(batch_size,fft_size,2,2).
        """
        assert len(mu_prior.shape) == 3
        self.bs = mu_prior.shape[0]
        self.fft_size = mu_prior.shape[1]
        assert mu_prior.shape[2] == 2
        assert cov_prior.shape == (self.bs, self.fft_size,2,2)
        assert not (t.diagonal(cov_prior, dim1=-2,dim2=-1) <= 0).any()
        self.mu_prior = mu_prior
        self.cov_prior = cov_prior
        self.prec_prior = t.inverse(cov_prior)
        self.info_prior = t.linalg.solve(cov_prior, mu_prior)

        self.dev = dev
        if precision == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif precision == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {precision}.")

        # Initialize lambda and gamma
        self.lam = t.diag_embed(1e-24*t.ones(size=(self.bs, self.fft_size, 2), dtype=self.prec))
        self.gam = t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec)

    def incorporate(self, info_cav, prec_cav):
        """
        Given complex Gaussian (cavity) distributions with mean mu_cav and 2x2 covariance matrix cov_cav,
        and independent Gaussian priors with mean mu_prior and covariance cov_prior,
        compute the moments of the resulting Gaussian as the product of both input distributions.
        :param t.tensor mu_cav: Mean of the cavity distribution. Shape=(batch_size,fft_size,2). Last dim is real/imag part. # TODO change to info/prec
        :param t.tensor cov_cav: Covariance matrices of the cavity distribution in each (complex) dimension. Shape=(batch_size,fft_size,2,2). # TODO change to info/prec
        :return:
            - mu_out    - Mean of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2).
            - cov_out   - Covariance of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2,2).
        """
        assert info_cav.shape == (self.bs, self.fft_size, 2) and info_cav.dtype == self.prec
        assert prec_cav.shape == (self.bs, self.fft_size, 2, 2)

        # Multiplication of 2 complex Gaussians
        prec_out = prec_cav + self.prec_prior
        info_out = info_cav + self.info_prior
        assert not (t.isinf(prec_out).any() or t.isnan(prec_out).any())
        assert not (t.isinf(info_out).any() or t.isnan(info_out).any())

        return info_out, prec_out

class EPFFTindependentGaussianMixture:
    """
    Class defining a fully factorized Gaussian mixture (GM) distribution.
    This can be used as input argument (prior_t or prior_f) to the EPFFT class.
    """
    def __init__(self, GM_weights, mu_GM_components, cov_GM_components, dev: t.device = t.device('cpu'), precision='double'):
        """
        :param t.tensor GM_weights: Weights of the Gaussian mixture components that sum to 1. Shape =(bs, fft_size, n_components).
        :param t.tensor mu_GM_components: Mean of the GM components of the GM prior distribution. Shape=(bs,fft_size, n_components,2). Last dim is real/imag part.
        :param t.tensor cov_GM_components: Covariances of the GM componenbts of the GM prior distribution. Shape=(bs,fft_size,n_components,2,2).
                                   Note that the variance for the real/imag part is var_prior/2, respectively.
        :param dev:
        :param precision:
        """
        assert len(GM_weights.shape) == 3
        self.bs = GM_weights.shape[0]
        self.fft_size = GM_weights.shape[1]
        self.n_components = GM_weights.shape[2]
        self.GM_weights = GM_weights

        assert len(mu_GM_components.shape) == 4
        assert mu_GM_components.shape == (self.bs, self.fft_size, self.n_components, 2)
        self.mu_GM_components = t.clone(mu_GM_components)
        assert cov_GM_components.shape == (self.bs, self.fft_size, self.n_components, 2,2)
        self.cov_GM_components = t.clone(cov_GM_components)
        self.info_GM_components = t.linalg.solve(cov_GM_components, mu_GM_components)
        self.prec_GM_components = t.inverse(cov_GM_components)

        self.dev = dev
        if precision == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif precision == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {precision}.")

        # Initialize lambda and gamma
        self.lam = t.diag_embed(1e-24*t.ones(size=(self.bs, self.fft_size, 2), dtype=self.prec))
        self.gam = t.zeros(size=(self.bs, self.fft_size, 2), dtype=self.prec)

    def incorporate(self, info_cav, prec_cav):
        """
        Given complex Gaussian (cavity) distributions with mean mu_cav and 2x2 covariance matrix cov_cav,
        and independent complex GM priors, compute the moments of the resulting GM as the product of both input distributions.
        :param t.tensor info_cav: Information vector of the cavity distribution. Shape=(batch_size,fft_size,2). Last dim is real/imag part.
        :param t.tensor prec_cav: Precision matrices of the cavity distribution in each (complex) dimension. Shape=(batch_size,fft_size,2,2).
        :return:
            - info_cav    - Mean of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2).
            - prec_cav   - Covariance of the incorporated Gaussian distribution. Shape=(batch_size,fft_size,2,2).
        """
        assert info_cav.shape == (self.bs, self.fft_size, 2) and info_cav.dtype == self.prec
        assert prec_cav.shape == (self.bs, self.fft_size, 2, 2)
        cov_cav = t.inverse(prec_cav)
        mu_cav = t.linalg.solve(prec_cav, info_cav)

        # 1.) Multiply each GM component with the Gaussian distribution -> again a GM with different weights and moments
        # new weights:
        weights_log = (t.log(self.GM_weights / t.sqrt(t.linalg.det(cov_cav.unsqueeze(2) + self.cov_GM_components))) + (-0.5 * t.sum((mu_cav.unsqueeze(2)-self.mu_GM_components) *
                    t.bmm(t.inverse(cov_cav.unsqueeze(2) + self.cov_GM_components).view(-1,2,2), (mu_cav.unsqueeze(2)-self.mu_GM_components).view(-1,2,1)).view(self.bs, self.fft_size, self.n_components, 2), dim=-1)))
        weights = t.exp(weights_log - t.logsumexp(weights_log, dim=-1, keepdim=True)) # shape (bs, fft_size, n_components)
        # new mean/cov of components
        cov_components = t.inverse(prec_cav.unsqueeze(2) + self.prec_GM_components)
        mu_component = t.bmm(cov_components.view(-1,2,2), (info_cav.unsqueeze(2) + self.info_GM_components).reshape(-1,2,1)).view(self.bs, self.fft_size, self.n_components, 2)
        # 2.) Compute moments of the new GM
        mu_out = t.sum(weights.unsqueeze(-1) * mu_component, dim=-2)
        cov_out = (t.sum(weights[...,None,None] * cov_components, dim=-3) +
                   t.sum(weights[...,None,None] * (t.bmm(mu_component.view(-1,2,1), mu_component.view(-1,1,2)).view(self.bs,self.fft_size,self.n_components,2,2) - t.bmm(mu_out.view(-1,2,1), mu_out.view(-1,1,2)).view(self.bs,self.fft_size,1,2,2)), dim=-3))
        prec_out = t.inverse(cov_out)
        info_out = t.linalg.solve(cov_out, mu_out)
        return info_out, prec_out
