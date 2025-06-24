# Copyright 2024, 2025 Luca Schmid.
import numpy as np
import torch as t
from collections import namedtuple

class FFTBP:
    """
    Performs Gaussian Belief Propagation on an FFT factor graph.
    This code is part of the EP-FFT framework [1].

    [1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'',
    International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC),
    Surrey, UK, July 2025.
    """
    def __init__(self, N: int, dev: t.device = t.device('cpu'), prec: str = 'double', convergence_tol=1e-5):
        """
        Initialize an instance of the FFTBP class with specific FFT size N.
        :param int N: FFT size.
        :param t.device dev: Device on which torch.tensors are stored, e.g., t.device('cpu') or t.device('cuda').
        :param str prec: Either single precision 'float' or double precision 'double'.
        """
        if not isinstance(N, int): raise ValueError(f'Input argument N must be integer but has type {type(N)}.')
        self.N = N
        self.n_stages = int(np.log2(N))
        self.dev = dev

        if prec == 'double':
            self.prec = t.double
            self.cprec = t.cdouble
        elif prec == 'float':
            self.prec = t.float
            self.cprec = t.cfloat
        else:
            raise ValueError(f"Precision only accepts the keywords <float> or <double>, but received {prec}.")

        self.forward_wiring, self.backward_wiring = self._get_graph_wiring()
        self.out_wire_f = t.cat((t.arange(0, N, step=2), t.arange(1, N, step=2)))
        self.butterfly_trafos, self.twiddle_per_layer = self._get_buttefly_trafos()
        self.butterfly_trafos_inv = t.inverse(self.butterfly_trafos)
        self.BP_messages = namedtuple("BP_messages", "in_msgs_left_mu in_msgs_left_cov in_msgs_right_mu in_msgs_right_cov")
        self.conv_tol=convergence_tol
        self.converged_after_n_iters = -1

    def BP_flooding(self, prior_mu_t, prior_cov_t, prior_mu_f, prior_cov_f, max_iters,
                    verbose=False, init_messages=None, return_messages=False):
        """
        Performs Gaussian BP on the FFT Butterfly Graph using a flooding schedule, i.e., all messages are updated
        in parallel.
        :param prior_mu_t: Mean of the prior in time domain. Shape = (bs, N, 2). Last dimension is real/imaginary part.
        :param prior_cov_t: 2x2 covariance matrices of the priors in time domain. Shape = (bs, N, 2,2). Last 2 dims are real/imaginary part.
        :param prior_mu_f: Mean of the prior in frequency domain. Shape = (bs, N, 2). Last dimension is real/imaginary part.
        :param prior_cov_f: 2x2 covariance matrices of the priors in frequency domain. Shape = (bs, N, 2,2). Last 2 dims are real/imaginary part.
        :param max_iters: Maximum number of BP flooding message passing iterations. Early stopping if all messages have converged.
        :param verbose: If true, prints some statements, like convergence info.
        :return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f: Mean and covariance for each variable in time and frequency domain, respectively.
                Same shapes as mu/cov from prior inputs.
        """
        assert max_iters >= 0
        assert len(prior_mu_t.shape) == 3
        bs = prior_mu_t.shape[0]
        assert prior_mu_t.shape == (bs,self.N,2) and prior_mu_f.shape == (bs,self.N,2) and prior_cov_t.shape == (bs,self.N,2,2) and prior_cov_f.shape == (bs,self.N,2,2)

        # Initialize all messages
        if init_messages == None:
            in_msgs_left_mu = t.zeros(size=(self.n_stages, bs, self.N, 2), dtype=self.prec, device=self.dev)
            in_msgs_left_cov = t.diag_embed(1e20 * t.ones_like(in_msgs_left_mu), dim1=-2, dim2=-1)
            in_msgs_right_mu = t.zeros(size=(self.n_stages, bs, self.N, 2), dtype=self.prec, device=self.dev)
            in_msgs_right_cov = t.diag_embed(1e20 * t.ones_like(in_msgs_left_mu), dim1=-2, dim2=-1)
        else:
            assert (init_messages.in_msgs_left_mu.shape == (self.n_stages,bs,self.N,2)) and (init_messages.in_msgs_left_mu.dtype == self.prec)
            in_msgs_left_mu = t.clone(init_messages.in_msgs_left_mu)
            assert (init_messages.in_msgs_left_cov.shape == (self.n_stages,bs,self.N,2,2)) and (init_messages.in_msgs_left_cov.dtype == self.prec)
            in_msgs_left_cov = t.clone(init_messages.in_msgs_left_cov)
            assert (init_messages.in_msgs_right_mu.shape == (self.n_stages,bs,self.N,2)) and (init_messages.in_msgs_right_mu.dtype == self.prec)
            in_msgs_right_mu = t.clone(init_messages.in_msgs_right_mu)
            assert (init_messages.in_msgs_right_cov.shape == (self.n_stages,bs,self.N,2,2)) and (init_messages.in_msgs_right_cov.dtype == self.prec)
            in_msgs_right_cov = t.clone(init_messages.in_msgs_right_cov)

        # The prior messages are overwritten with the (new) prior information.
        in_msgs_left_mu[0] = t.clone(prior_mu_t[:,self.forward_wiring[0]]) # These messages will not change during the BP iterations.
        in_msgs_left_cov[0] = t.clone(prior_cov_t[:,self.forward_wiring[0]]) # These messages will not change during the BP iterations.
        in_msgs_right_mu[-1] = t.clone(prior_mu_f[:,self.backward_wiring[-1]]) # These messages will not change during the BP iterations.
        in_msgs_right_cov[-1] = t.clone(prior_cov_f[:,self.backward_wiring[-1]]) # These messages will not change during the BP iterations.

        for iter_i in range(max_iters):
            # Parallel message update at all butterfly factor nodes.
            (out_msgs_x_mu, out_msgs_x_cov, out_msgs_y_mu, out_msgs_y_cov) = (
                self._butterfly_message_update(mu_x1=in_msgs_left_mu[:, :, ::2].view(-1, 2), cov_x1=in_msgs_left_cov[:, :, ::2].view(-1, 2, 2),
                                               mu_x2=in_msgs_left_mu[:,:,1::2].view(-1,2), cov_x2=in_msgs_left_cov[:,:,1::2].view(-1,2,2),
                                               twiddle=self.twiddle_per_layer.view(self.n_stages,1,self.N//2).repeat(1,bs,1).flatten(),
                                               trafo_for=self.butterfly_trafos.view(self.n_stages,1,self.N//2,4,4).repeat(1,bs,1,1,1).view(-1,4,4),
                                               trafo_back=self.butterfly_trafos_inv.view(self.n_stages,1,self.N//2,4,4).repeat(1,bs,1,1,1).view(-1,4,4),
                                               mu_y1=in_msgs_right_mu[:,:,::2].view(-1,2), cov_y1=in_msgs_right_cov[:,:,::2].view(-1,2,2),
                                               mu_y2=in_msgs_right_mu[:,:,1::2].view(-1,2), cov_y2=in_msgs_right_cov[:,:,1::2].view(-1,2,2)))
            # Message passing.
            in_msgs_left_mu, in_msgs_left_cov, in_msgs_right_mu, in_msgs_right_cov, converged = (
                self._message_passing(out_msgs_x_mu=out_msgs_x_mu.view(self.n_stages,bs,self.N,2), out_msgs_x_cov=out_msgs_x_cov.view(self.n_stages,bs,self.N,2,2),
                                      out_msgs_y_mu=out_msgs_y_mu.view(self.n_stages,bs,self.N,2), out_msgs_y_cov=out_msgs_y_cov.view(self.n_stages,bs,self.N,2,2),
                                      in_msgs_left_mu_old=in_msgs_left_mu, in_msgs_left_cov_old=in_msgs_left_cov,
                                      in_msgs_right_mu_old=in_msgs_right_mu, in_msgs_right_cov_old=in_msgs_right_cov))

            if converged:
                if verbose:
                    print(f"Converged after {iter_i+1} flooding iterations.")
                self.converged_after_n_iters = iter_i
                break

        # final belief computation
        belief_mu_t, belief_cov_t = self._compute_beliefs_t(out_msg_x_mu=out_msgs_x_mu.view(self.n_stages,bs,self.N,2)[0], out_msg_x_cov=out_msgs_x_cov.view(self.n_stages,bs,self.N,2,2)[0],
                                                            in_msg_left_mu=in_msgs_left_mu[0], in_msg_left_cov=in_msgs_left_cov[0])
        belief_mu_f, belief_cov_f = self._compute_beliefs_f(out_msg_y_mu=out_msgs_y_mu.view(self.n_stages,bs,self.N,2)[-1], out_msg_y_cov=out_msgs_y_cov.view(self.n_stages,bs,self.N,2,2)[-1],
                                                            in_msg_right_mu=in_msgs_right_mu[-1], in_msg_right_cov=in_msgs_right_cov[-1])

        if return_messages:
            return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f, self.BP_messages(in_msgs_left_mu, in_msgs_left_cov, in_msgs_right_mu, in_msgs_right_cov)
        else:
            return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f

    def BP_layered(self, prior_mu_t, prior_cov_t, prior_mu_f, prior_cov_f, max_iters,
                   verbose=False, init_messages=None, return_messages=False):
        """
        Performs Gaussian BP on the FFT Butterfly Graph using a layered schedule:
        All outgoing messages from the N/2 butterflies in one FFT layer are updated in parallel.
        Starting in the time domain, the messages thereby propagate to the frequency domain and then back. One iteration
        is one forward+backward path.
        :param prior_mu_t: Mean of the prior in time domain. Shape = (bs, N, 2). Last dimension is real/imaginary part.
        :param prior_cov_t: 2x2 covariance matrices of the priors in time domain. Shape = (bs, N, 2,2). Last 2 dims are real/imaginary part.
        :param prior_mu_f: Mean of the prior in frequency domain. Shape = (bs, N, 2). Last dimension is real/imaginary part.
        :param prior_cov_f: 2x2 covariance matrices of the priors in frequency domain. Shape = (bs, N, 2,2). Last 2 dims are real/imaginary part.
        :param max_iters: Maximum number of BP flooding message passing iterations. Early stopping if all messages have converged.
        :param verbose: If true, prints some statements, like convergence info.
        :return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f: Mean and covariance for each variable in time and frequency domain, respectively.
                Same shapes as mu/cov from prior inputs.
        """
        assert max_iters >= 0
        assert len(prior_mu_t.shape) == 3
        bs = prior_mu_t.shape[0]
        assert prior_mu_t.shape == (bs,self.N,2) and prior_mu_f.shape == (bs,self.N,2) and prior_cov_t.shape == (bs,self.N,2,2) and prior_cov_f.shape == (bs,self.N,2,2)

        # Initialize all messages
        if init_messages == None:
            in_msgs_left_mu = t.zeros(size=(self.n_stages, bs, self.N, 2), dtype=self.prec, device=self.dev)
            in_msgs_left_cov = t.diag_embed(1e20 * t.ones_like(in_msgs_left_mu), dim1=-2, dim2=-1)
            in_msgs_right_mu = t.zeros(size=(self.n_stages, bs, self.N, 2), dtype=self.prec, device=self.dev)
            in_msgs_right_cov = t.diag_embed(1e20 * t.ones_like(in_msgs_left_mu), dim1=-2, dim2=-1)
        else:
            assert (init_messages.in_msgs_left_mu.shape == (self.n_stages,bs,self.N,2)) and (init_messages.in_msgs_left_mu.dtype == self.prec)
            in_msgs_left_mu = t.clone(init_messages.in_msgs_left_mu)
            assert (init_messages.in_msgs_left_cov.shape == (self.n_stages,bs,self.N,2,2)) and (init_messages.in_msgs_left_cov.dtype == self.prec)
            in_msgs_left_cov = t.clone(init_messages.in_msgs_left_cov)
            assert (init_messages.in_msgs_right_mu.shape == (self.n_stages,bs,self.N,2)) and (init_messages.in_msgs_right_mu.dtype == self.prec)
            in_msgs_right_mu = t.clone(init_messages.in_msgs_right_mu)
            assert (init_messages.in_msgs_right_cov.shape == (self.n_stages,bs,self.N,2,2)) and (init_messages.in_msgs_right_cov.dtype == self.prec)
            in_msgs_right_cov = t.clone(init_messages.in_msgs_right_cov)

        # The prior messages are overwritten with the (new) prior information.
        in_msgs_left_mu[0] = t.clone(prior_mu_t[:,self.forward_wiring[0]]) # These messages will not change during the BP iterations.
        in_msgs_left_cov[0] = t.clone(prior_cov_t[:,self.forward_wiring[0]]) # These messages will not change during the BP iterations.
        in_msgs_right_mu[-1] = t.clone(prior_mu_f[:,self.backward_wiring[-1]]) # These messages will not change during the BP iterations.
        in_msgs_right_cov[-1] = t.clone(prior_cov_f[:,self.backward_wiring[-1]]) # These messages will not change during the BP iterations.

        for iter_i in range(max_iters):
            in_msgs_left_mu_old = t.clone(in_msgs_left_mu)
            in_msgs_left_cov_old = t.clone(in_msgs_left_cov)
            in_msgs_right_mu_old = t.clone(in_msgs_right_mu)
            in_msgs_right_cov_old = t.clone(in_msgs_right_cov)
            # Update all nodes in one FFT layer in parallel (go from time domain to frequency domain and back)
            for layer_i in list(range(self.n_stages)) + list(range(self.n_stages-2,-1,-1)):
                (out_msgs_x_mu, out_msgs_x_cov, out_msgs_y_mu, out_msgs_y_cov) = (
                    self._butterfly_message_update(mu_x1=in_msgs_left_mu[layer_i,:,::2].view(-1, 2), cov_x1=in_msgs_left_cov[layer_i,:,::2].view(-1, 2, 2),
                                                   mu_x2=in_msgs_left_mu[layer_i,:,1::2].view(-1,2), cov_x2=in_msgs_left_cov[layer_i,:,1::2].view(-1,2,2),
                                                   twiddle=self.twiddle_per_layer[layer_i].view(1,self.N//2).repeat(bs,1).flatten(),
                                                   trafo_for=self.butterfly_trafos[layer_i].view(1,self.N//2,4,4).repeat(bs,1,1,1).view(-1,4,4),
                                                   trafo_back=self.butterfly_trafos_inv[layer_i].view(1,self.N//2,4,4).repeat(bs,1,1,1).view(-1,4,4),
                                                   mu_y1=in_msgs_right_mu[layer_i,:,::2].view(-1,2), cov_y1=in_msgs_right_cov[layer_i,:,::2].view(-1,2,2),
                                                   mu_y2=in_msgs_right_mu[layer_i,:,1::2].view(-1,2), cov_y2=in_msgs_right_cov[layer_i,:,1::2].view(-1,2,2)))

                if layer_i == self.n_stages-1: # save out_msgs_y here for belief_f computation
                    out_msg_y_mu_laststage = t.clone(out_msgs_y_mu)
                    out_msg_y_cov_laststage = t.clone(out_msgs_y_cov)

                # message passing of out messages.
                if layer_i > 0:
                    in_msgs_right_mu[layer_i-1] = t.clone(out_msgs_x_mu.view(bs,self.N,2)[:, self.backward_wiring[layer_i-1]])
                    in_msgs_right_cov[layer_i-1]= t.clone(out_msgs_x_cov.view(bs,self.N,2,2)[:,self.backward_wiring[layer_i-1]])

                if layer_i < self.n_stages-1:
                    in_msgs_left_mu[layer_i+1] = t.clone(out_msgs_y_mu.view(bs,self.N,2)[:, self.forward_wiring[layer_i+1]])
                    in_msgs_left_cov[layer_i+1]= t.clone(out_msgs_y_cov.view(bs,self.N,2,2)[:,self.forward_wiring[layer_i+1]])


            # Check for convergence
            in_msgs_left_mu_converged = (t.abs(in_msgs_left_mu - in_msgs_left_mu_old) < self.conv_tol * (1 + t.abs(in_msgs_left_mu_old))).all()
            in_msgs_left_cov_converged = (t.abs(in_msgs_left_cov - in_msgs_left_cov_old) < self.conv_tol * (1 + t.abs(in_msgs_left_cov_old))).all()
            in_msgs_right_mu_converged = (t.abs(in_msgs_right_mu - in_msgs_right_mu_old) < self.conv_tol * (1 + t.abs(in_msgs_right_mu_old))).all()
            in_msgs_right_cov_converged = (t.abs(in_msgs_right_cov - in_msgs_right_cov_old) < self.conv_tol * (1 + t.abs(in_msgs_right_cov_old))).all()
            converged = in_msgs_left_mu_converged and in_msgs_left_cov_converged and in_msgs_right_mu_converged and in_msgs_right_cov_converged

            if converged:
                if verbose:
                    print(f"Converged after {iter_i+1} layered iterations.")
                self.converged_after_n_iters = iter_i
                break

        # final belief computation
        belief_mu_t, belief_cov_t = self._compute_beliefs_t(out_msg_x_mu=out_msgs_x_mu.view(bs,self.N,2), out_msg_x_cov=out_msgs_x_cov.view(bs,self.N,2,2),
                                                            in_msg_left_mu=in_msgs_left_mu[0], in_msg_left_cov=in_msgs_left_cov[0])
        belief_mu_f, belief_cov_f = self._compute_beliefs_f(out_msg_y_mu=out_msg_y_mu_laststage.view(bs,self.N,2), out_msg_y_cov=out_msg_y_cov_laststage.view(bs,self.N,2,2),
                                                            in_msg_right_mu=in_msgs_right_mu[-1], in_msg_right_cov=in_msgs_right_cov[-1])

        if return_messages:
            return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f, self.BP_messages(in_msgs_left_mu, in_msgs_left_cov, in_msgs_right_mu, in_msgs_right_cov)
        else:
            return belief_mu_t, belief_cov_t, belief_mu_f, belief_cov_f

    def _compute_beliefs_t(self, out_msg_x_mu, out_msg_x_cov, in_msg_left_mu, in_msg_left_cov):
        assert len(out_msg_x_mu.shape) == 3
        bs = out_msg_x_mu.shape[0]
        assert out_msg_x_mu.shape == (bs,self.N,2) and out_msg_x_cov.shape == (bs,self.N,2,2)
        assert in_msg_left_mu.shape == (bs, self.N, 2) and in_msg_left_cov.shape == (bs, self.N, 2, 2)
        belief_prec_t = t.inverse(in_msg_left_cov) + t.inverse(out_msg_x_cov)
        belief_cov_t = t.inverse(belief_prec_t)
        belief_info_t = t.linalg.solve(in_msg_left_cov, in_msg_left_mu) + t.linalg.solve(out_msg_x_cov, out_msg_x_mu)
        belief_mu_t = t.linalg.solve(belief_prec_t, belief_info_t)
        return belief_mu_t[:,self.forward_wiring[0]], belief_cov_t[:,self.forward_wiring[0]]

    def _compute_beliefs_f(self, out_msg_y_mu, out_msg_y_cov, in_msg_right_mu, in_msg_right_cov):
        assert len(out_msg_y_mu.shape) == 3
        bs = out_msg_y_mu.shape[0]
        assert out_msg_y_mu.shape == (bs,self.N,2) and out_msg_y_cov.shape == (bs,self.N,2,2)
        assert in_msg_right_mu.shape == (bs, self.N, 2) and in_msg_right_cov.shape == (bs, self.N, 2, 2)
        belief_prec_f = t.inverse(out_msg_y_cov) + t.inverse(in_msg_right_cov)
        belief_cov_f = t.inverse(belief_prec_f)
        belief_info_f = t.linalg.solve(out_msg_y_cov, out_msg_y_mu) + t.linalg.solve(in_msg_right_cov, in_msg_right_mu)
        belief_mu_f = t.linalg.solve(belief_prec_f, belief_info_f)
        return belief_mu_f[:, self.out_wire_f], belief_cov_f[:, self.out_wire_f]

    def _butterfly_message_update(self, mu_x1, cov_x1, mu_x2, cov_x2, twiddle, trafo_for, trafo_back, mu_y1, cov_y1, mu_y2, cov_y2):
        """
        Given the Gaussian input messages (specified by mean and covariance) at all 4 ports of a 2x2 butterfly node,
        apply a BP message update for each outgoing message, respectively.
        A butterfly node is defined as the linear relationship y1=x1+wx2 and y2=x1-wx2 where w is the complex-valued twiddle factor.
        Conventional message update with linear transform of 2 variables to the other side...
        See eq.(7) and Fig. 2 in [1] for more details.
        :param t.tensor mu_x1: Mean of incident message at port x1 of shape (bs, 2).
        :param t.tensor cov_x1: Covariance of incident message at port x1 of shape (bs, 2, 2).
        :param t.tensor mu_x2: Mean of incident message at port x2 of shape (bs, 2).
        :param t.tensor cov_x2: Covariance of incident message at port x2 of shape (bs, 2, 2).
        :param t.tensor twiddle: Complex-valued twiddle factor w of shape (bs,).
        :param t.tensor mu_y1: Mean of incident message at port y1 of shape (bs, 2).
        :param t.tensor cov_y1: Covariance of incident message at port y1 of shape (bs, 2, 2).
        :param t.tensor mu_y2: Mean of incident message at port y2 of shape (bs, 2).
        :param t.tensor cov_y2: Covariance of incident message at port y2 of shape (bs, 2, 2).
        :return: mu_x_out (shape = (bs,2,2) where [:,0,:] is the output for x1 and [:,1,:] is the output for x2,
                 cov_x_out (shape= (bs,2,2,2) where [:,0,:,:] is the output for x1 and [:,1,:,:] is the output for x2,
                 mu_y_out, cov_y_out (same principle as for x).
        """
        assert len(mu_x1.shape) == 2
        bs = mu_x1.shape[0] # batch_size
        assert mu_x1.shape == (bs, 2) and mu_x2.shape == (bs, 2) and mu_y1.shape == (bs, 2) and mu_y2.shape == (bs, 2)
        assert cov_x1.shape == (bs, 2, 2) and cov_x2.shape == (bs, 2, 2) and cov_y1.shape == (bs, 2, 2) and cov_y2.shape == (bs, 2, 2)
        assert twiddle.shape == (bs,) and twiddle.dtype == self.cprec

        # Compute outputs for y1 and y2.
        mu_y1_out, cov_y1_out, mu_y2_out, cov_y2_out = (
            self._onesided_butterfly_message_update(mu1_in=mu_x1, cov1_in=cov_x1, mu2_in=mu_x2, cov2_in=cov_x2,
                                                    w=twiddle, trafo=trafo_for, trafo_inv=trafo_back,
                                                    mu1_out=mu_y1, cov1_out=cov_y1, mu2_out=mu_y2, cov2_out=cov_y2))
        # Compute outputs for x1 and x2.
        mu_x1_out, cov_x1_out, mu_x2_out, cov_x2_out = (
            self._onesided_butterfly_message_update(mu1_in=mu_y1, cov1_in=cov_y1, mu2_in=mu_y2, cov2_in=cov_y2,
                                                    w=twiddle, trafo=trafo_back, trafo_inv=trafo_for,
                                                    mu1_out=mu_x1, cov1_out=cov_x1, mu2_out=mu_x2, cov2_out=cov_x2))

        return t.stack((mu_x1_out, mu_x2_out), dim=-2), t.stack((cov_x1_out, cov_x2_out), dim=-3), t.stack((mu_y1_out, mu_y2_out), dim=-2), t.stack((cov_y1_out, cov_y2_out), dim=-3)

    def _onesided_butterfly_message_update(self, mu1_in, cov1_in, mu2_in, cov2_in, w, trafo, trafo_inv, mu1_out, cov1_out, mu2_out, cov2_out):
        """
        One-sided butterfly message update. See eq.(7) and Fig. 2 in [1] for more details.
        """
        assert len(mu1_in.shape) == 2
        bs = mu1_in.shape[0] # batch_size
        assert mu1_in.shape[1] == 2
        assert mu1_in.shape == mu2_in.shape
        assert len(cov1_in.shape) == 3
        assert cov1_in.shape[1:] == (2, 2)
        assert cov1_in.shape == cov2_in.shape
        assert trafo_inv.shape == (bs, 4,4)
        assert trafo.shape == trafo_inv.shape

        assert mu1_in.shape == mu1_out.shape and mu1_in.shape == mu2_out.shape
        assert cov1_in.shape == cov1_out.shape and cov1_in.shape == cov2_out.shape

        # create different masks to handle different cases
        i1_0 = t.sum(t.diagonal(cov1_in, dim1=-2,dim2=-1) < 1e-8, dim=-1).to(t.bool)
        i1_inf = t.sum(t.diagonal(cov1_in, dim1=-2, dim2=-1) > 1e8, dim=-1).to(t.bool)
        i1_norm = t.logical_and(~i1_0, ~i1_inf)
        i2_0 = t.sum(t.diagonal(cov2_in, dim1=-2, dim2=-1) < 1e-8, dim=-1).to(t.bool)
        i2_inf = t.sum(t.diagonal(cov2_in, dim1=-2, dim2=-1) > 1e8, dim=-1).to(t.bool)
        i2_norm = t.logical_and(~i2_0, ~i2_inf)
        o1_0 = t.sum(t.diagonal(cov1_out, dim1=-2, dim2=-1) < 1e-8, dim=-1).to(t.bool)
        o1_inf = t.sum(t.diagonal(cov1_out, dim1=-2, dim2=-1) > 1e8, dim=-1).to(t.bool)
        o1_norm = t.logical_and(~o1_0, ~o1_inf)
        o2_0 = t.sum(t.diagonal(cov2_out, dim1=-2, dim2=-1) < 1e-8, dim=-1).to(t.bool)
        o2_inf = t.sum(t.diagonal(cov2_out, dim1=-2, dim2=-1) > 1e8, dim=-1).to(t.bool)
        o2_norm = t.logical_and(~o2_0, ~o2_inf)

        lam_mask = ~t.logical_and(i1_inf, i2_inf)
        d_mask = ~t.logical_and(i1_0, i2_0)

        # For all elements in lam_mask, compute canonical form of linear transformation (in order to incorporate info from one output port)
        B = t.cat((t.cat((t.bmm(t.transpose(trafo_inv[lam_mask,:2,:2], dim0=-2,dim1=-1), t.inverse(cov1_in[lam_mask])), t.bmm(t.transpose(trafo_inv[lam_mask,2:,:2], dim0=-2, dim1=-1), t.inverse(cov2_in[lam_mask]))), dim=-1),
                   t.cat((t.bmm(t.transpose(trafo_inv[lam_mask,:2,2:], dim0=-2, dim1=-1), t.inverse(cov1_in[lam_mask])), t.bmm(t.transpose(trafo_inv[lam_mask,2:,2:], dim0=-2, dim1=-1), t.inverse(cov2_in[lam_mask]))), dim=-1)), dim=-2) # B = ((A_w)^{-1})^H * block_diag(cov1_left, cov2_left)
        lam = t.bmm(B, trafo_inv[lam_mask])
        gam = t.bmm(B, t.cat((mu1_in[lam_mask], mu2_in[lam_mask]), dim=-1).unsqueeze(-1)).view(-1, 4)

        # For all elements in d_mask, compute direct linear transformation from input to output
        mu_out_direct = t.bmm(trafo[d_mask], t.cat((mu1_in[d_mask], mu2_in[d_mask]), dim=-1).unsqueeze(-1)).view(-1, 4)
        cov_out_direct = t.bmm(t.cat((t.cat((t.bmm(trafo[d_mask,:2,:2], cov1_in[d_mask]), t.bmm(trafo[d_mask,:2,2:], cov2_in[d_mask])), dim=-1),
                               t.cat((t.bmm(trafo[d_mask,2:,:2], cov1_in[d_mask]), t.bmm(trafo[d_mask,2:,2:], cov2_in[d_mask])), dim=-1)), dim=-2), t.transpose(trafo[d_mask], dim0=-2, dim1=-1))


        # Compute result for o1 (incorporating information from o2)
        mu1_res = t.zeros_like(mu1_in)  # init
        cov1_res = t.zeros_like(cov1_in)  # init
        # Case 1: Compute via lambda
        o1_lam_mask = t.logical_or(t.logical_and(i1_0, i2_0), t.logical_or(t.logical_and(t.logical_or(t.logical_and(i1_0, i2_inf), t.logical_and(i1_inf, i2_0)), o2_0), t.logical_and(t.logical_or(i1_norm, i2_norm), t.logical_and(~o2_inf, ~t.logical_and(t.logical_or(i1_0, i2_0), o2_norm)))))
        lam2 = t.clone(lam[o1_lam_mask[lam_mask]])
        lam2[:,2:,2:] += t.inverse(cov2_out[o1_lam_mask])
        cov1_res_full = t.inverse(lam2)
        cov1_res[o1_lam_mask] = cov1_res_full[:,:2,:2]
        gam2 = t.clone(gam[o1_lam_mask[lam_mask]])
        gam2[:,2:] += t.linalg.solve(cov2_out[o1_lam_mask], mu2_out[o1_lam_mask])
        mu1_res[o1_lam_mask] = t.linalg.solve(lam2, gam2)[:,:2]
        # Case 2: Compute directly (no significant info from o2 to incorporate)
        o1_d_mask = t.logical_and(~t.logical_and(i1_0, i2_0), t.logical_or(t.logical_and(i1_inf, i2_inf), o2_inf))
        mu1_res[o1_d_mask] = mu_out_direct[o1_d_mask[d_mask]][:,:2]
        cov1_res[o1_d_mask] = cov_out_direct[o1_d_mask[d_mask]][:,:2,:2]
        # Case 3: special case, where one of the inputs has zero variance, the other input has inf variance and o2 has finite variance
        # 3.1 i1 is known, i2 is unknown and o2 is finite
        o1_i10i2inf_mask = t.logical_and(t.logical_and(i1_0, i2_inf), t.logical_and(~o2_0, ~o2_inf))
        mu1_res[o1_i10i2inf_mask] = 2*mu1_in[o1_i10i2inf_mask] - mu2_out[o1_i10i2inf_mask]
        cov1_res[o1_i10i2inf_mask] = t.clone(cov2_out[o1_i10i2inf_mask])
        # 3.2 i2 is known, i1 is unknown and o2 is finite
        o1_i20i1inf_mask = t.logical_and(t.logical_and(i1_inf, i2_0), t.logical_and(~o2_0, ~o2_inf))
        mu1_res[o1_i20i1inf_mask] = 2*t.view_as_real(w[o1_i20i1inf_mask]*t.view_as_complex(mu2_in[o1_i20i1inf_mask])) + mu2_out[o1_i20i1inf_mask]
        cov1_res[o1_i20i1inf_mask] = t.clone(cov2_out[o1_i20i1inf_mask])
        # Case 4: special case, where one of the inputs has zero variance, the other input has finite variance and o2 has finite variance
        # 4.1 i1 is known, i2 has finite variance, and o2 is finite
        o1_i10_mask = t.logical_and(o2_norm, t.logical_and(i1_0, i2_norm))
        o1_i10_lam = t.inverse(cov_out_direct[o1_i10_mask[d_mask],:2,:2]) + t.inverse(cov2_out[o1_i10_mask])
        cov1_res[o1_i10_mask] = t.inverse(o1_i10_lam)
        mu1_res[o1_i10_mask] = t.linalg.solve(o1_i10_lam, t.linalg.solve(cov_out_direct[o1_i10_mask[d_mask],:2,:2], mu_out_direct[o1_i10_mask[d_mask],:2]) + t.linalg.solve(cov2_out[o1_i10_mask], 2*mu1_in[o1_i10_mask]-mu2_out[o1_i10_mask]))
        # 4.2 i1 has finite variance, i2 is known, and o2 is finite
        o1_i20_mask = t.logical_and(o2_norm, t.logical_and(i1_norm, i2_0))
        o1_i20_lam = t.inverse(cov_out_direct[o1_i20_mask[d_mask],:2,:2]) + t.inverse(cov2_out[o1_i20_mask])
        cov1_res[o1_i20_mask] = t.inverse(o1_i20_lam)
        mu1_res[o1_i20_mask] = t.linalg.solve(o1_i20_lam, t.linalg.solve(cov_out_direct[o1_i20_mask[d_mask],:2,:2], mu_out_direct[o1_i20_mask[d_mask],:2]) + t.linalg.solve(cov2_out[o1_i20_mask], 2*t.view_as_real(w[o1_i20_mask]*t.view_as_complex(mu2_in[o1_i20_mask]))-mu2_out[o1_i20_mask]))

        assert ((o1_lam_mask.int() + o1_d_mask.int() + o1_i10i2inf_mask.int() + o1_i20i1inf_mask.int() + o1_i10_mask.int() + o1_i20_mask.int()) == 1).all() # assert coverage


        # Compute result for o2 (incorporating information from o1)
        mu2_res = t.zeros_like(mu1_in)  # init
        cov2_res = t.zeros_like(cov1_in)  # init
        # Case 1: Compute via lambda
        o2_lam_mask = t.logical_or(t.logical_and(i1_0, i2_0), t.logical_or(t.logical_and(t.logical_or(t.logical_and(i1_0, i2_inf), t.logical_and(i1_inf, i2_0)), o2_0), t.logical_and(t.logical_or(i1_norm, i2_norm), t.logical_and(~o1_inf, ~t.logical_and(t.logical_or(i1_0, i2_0), o1_norm)))))
        lam1 = t.clone(lam[o2_lam_mask[lam_mask]])
        lam1[:,:2,:2] += t.inverse(cov1_out[o2_lam_mask])
        cov2_res_full = t.inverse(lam1)
        cov2_res[o2_lam_mask] = cov2_res_full[:,2:,2:]
        gam1 = t.clone(gam[o2_lam_mask[lam_mask]])
        gam1[:,:2] += t.linalg.solve(cov1_out[o2_lam_mask], mu1_out[o2_lam_mask])
        mu2_res[o2_lam_mask] = t.linalg.solve(lam1, gam1)[:,2:]
        # Case 2: Compute directly (no significant info from o1 to incorporate)
        o2_d_mask = t.logical_and(~t.logical_and(i1_0, i2_0), t.logical_or(t.logical_and(i1_inf, i2_inf), o1_inf))
        mu2_res[o2_d_mask] = mu_out_direct[o2_d_mask[d_mask]][:,2:]
        cov2_res[o2_d_mask] = cov_out_direct[o2_d_mask[d_mask]][:,2:,2:]
        # Case 3: special case, where one of the inputs has zero variance and o1 has finite variance
        # 3.1 i1 is known, i2 is unknown and o1 is finite
        o2_i10i2inf_mask = t.logical_and(t.logical_and(i1_0, i2_inf), t.logical_and(~o1_0, ~o1_inf))
        mu2_res[o2_i10i2inf_mask] = 2*mu1_in[o2_i10i2inf_mask] - mu1_out[o2_i10i2inf_mask]
        cov2_res[o2_i10i2inf_mask] = t.clone(cov1_out[o2_i10i2inf_mask])
        # 3.2 i2 is known, i1 is unknown and o1 is finite
        o2_i20i1inf_mask = t.logical_and(t.logical_and(i1_inf, i2_0), t.logical_and(~o1_0, ~o1_inf))
        mu2_res[o2_i20i1inf_mask] = -2*t.view_as_real(w[o2_i20i1inf_mask]*t.view_as_complex(mu2_in[o2_i20i1inf_mask])) + mu1_out[o2_i20i1inf_mask]
        cov2_res[o2_i20i1inf_mask] = t.clone(cov1_out[o2_i20i1inf_mask])
        # Case 4: special case, where one of the inputs has zero variance, the other input has finite variance and o1 has finite variance
        # 4.1 i1 is known, i2 has finite variance, and o1 is finite
        o2_i10_mask = t.logical_and(o1_norm, t.logical_and(i1_0, i2_norm))
        o2_i10_lam = t.inverse(cov_out_direct[o2_i10_mask[d_mask],2:,2:]) + t.inverse(cov1_out[o2_i10_mask])
        cov2_res[o2_i10_mask] = t.inverse(o2_i10_lam)
        mu2_res[o2_i10_mask] = t.linalg.solve(o2_i10_lam, t.linalg.solve(cov_out_direct[o2_i10_mask[d_mask],2:,2:], mu_out_direct[o2_i10_mask[d_mask],2:]) + t.linalg.solve(cov1_out[o2_i10_mask], 2*mu1_in[o2_i10_mask]-mu1_out[o2_i10_mask]))
        # 4.2 i1 has finite variance, i2 is known, and o1 is finite
        o2_i20_mask = t.logical_and(o1_norm, t.logical_and(i1_norm, i2_0))
        o2_i20_lam = t.inverse(cov_out_direct[o2_i20_mask[d_mask],2:,2:]) + t.inverse(cov1_out[o2_i20_mask])
        cov2_res[o2_i20_mask] = t.inverse(o2_i20_lam)
        mu2_res[o2_i20_mask] = t.linalg.solve(o2_i20_lam, t.linalg.solve(cov_out_direct[o2_i20_mask[d_mask],2:,2:], mu_out_direct[o2_i20_mask[d_mask],2:]) + t.linalg.solve(cov1_out[o2_i20_mask], -2*t.view_as_real(w[o2_i20_mask]*t.view_as_complex(mu2_in[o1_i20_mask]))-mu1_out[o2_i20_mask]))
        assert ((o2_lam_mask.int() + o2_d_mask.int() + o2_i10i2inf_mask.int() + o2_i20i1inf_mask.int() + o2_i10_mask.int() + o2_i20_mask.int()) == 1).all() # assert coverage

        return mu1_res, cov1_res, mu2_res, cov2_res

    def _message_passing(self, out_msgs_x_mu, out_msgs_x_cov, out_msgs_y_mu, out_msgs_y_cov, in_msgs_left_mu_old, in_msgs_left_cov_old, in_msgs_right_mu_old, in_msgs_right_cov_old):
        """
        Applies the FFT graph wiring to pass the output messages to their new inputs.
        See Fig. 1 in [1] for more details.
        :param out_msgs_x_mu:
        :param out_msgs_x_cov:
        :param out_msgs_y_mu:
        :param out_msgs_y_cov:
        :param in_msgs_left_mu_old:
        :param in_msgs_left_cov_old:
        :param in_msgs_right_mu_old:
        :param in_msgs_right_cov:
        :return:
        """
        assert len(out_msgs_x_mu.shape) == 4
        bs = out_msgs_x_mu.shape[1]
        assert out_msgs_x_mu.shape == (self.n_stages,bs,self.N,2) and out_msgs_x_cov.shape == (self.n_stages,bs,self.N,2,2) and out_msgs_y_mu.shape == (self.n_stages,bs,self.N,2) and out_msgs_y_cov.shape == (self.n_stages,bs,self.N,2,2)
        assert in_msgs_left_mu_old.shape == (self.n_stages,bs,self.N,2) and in_msgs_left_cov_old.shape == (self.n_stages,bs,self.N,2,2) and in_msgs_right_mu_old.shape == (self.n_stages,bs,self.N,2) and in_msgs_right_cov_old.shape == (self.n_stages,bs,self.N,2,2)

        in_msgs_left_mu_new = t.clone(in_msgs_left_mu_old)
        in_msgs_left_cov_new = t.clone(in_msgs_left_cov_old)
        in_msgs_right_mu_new = t.clone(in_msgs_right_mu_old)
        in_msgs_right_cov_new = t.clone(in_msgs_right_cov_old)

        for stage_i in range(1, self.n_stages): # Propagate output  messages to new input messages
            in_msgs_left_mu_new[stage_i] = out_msgs_y_mu.view(self.n_stages,bs,self.N,2)[stage_i-1,:,self.forward_wiring[stage_i]]
            in_msgs_left_cov_new[stage_i] = out_msgs_y_cov.view(self.n_stages,bs,self.N,2,2)[stage_i-1,:,self.forward_wiring[stage_i]]
            in_msgs_right_mu_new[stage_i-1] = out_msgs_x_mu.view(self.n_stages,bs,self.N,2)[stage_i,:,self.backward_wiring[stage_i-1]]
            in_msgs_right_cov_new[stage_i-1] = out_msgs_x_cov.view(self.n_stages,bs,self.N,2,2)[stage_i,:,self.backward_wiring[stage_i-1]]

        # Check for convergence
        in_msgs_left_mu_converged = (t.abs(in_msgs_left_mu_new - in_msgs_left_mu_old) < self.conv_tol * (1 + t.abs(in_msgs_left_mu_old))).all()
        in_msgs_left_cov_converged = (t.abs(in_msgs_left_cov_new - in_msgs_left_cov_old) < self.conv_tol * (1 + t.abs(in_msgs_left_cov_old))).all()
        in_msgs_right_mu_converged = (t.abs(in_msgs_right_mu_new - in_msgs_right_mu_old) < self.conv_tol * (1 + t.abs(in_msgs_right_mu_old))).all()
        in_msgs_right_cov_converged = (t.abs(in_msgs_right_cov_new - in_msgs_right_cov_old) < self.conv_tol * (1 + t.abs(in_msgs_right_cov_old))).all()
        converged = in_msgs_left_mu_converged and in_msgs_left_cov_converged and in_msgs_right_mu_converged and in_msgs_right_cov_converged

        return in_msgs_left_mu_new, in_msgs_left_cov_new, in_msgs_right_mu_new, in_msgs_right_cov_new, converged

    def _get_graph_wiring(self):
        """
        Collects the wiring between the butterfly stages.
        For the forward_wiring, the first layer is the reversed bit ordering. The rest are the wiring
        between the butterfly stages. The backward_wiring contains the same wirings of forward_wiring[1:-1] at [0:-2].
        The last element of backward_wiring is reordering after the last butterfly stage.
        :return: 2 Tensor (forward_wiring, backward_wiring) of shape (self.n_stages, N) with self.n_stages = log2(N).
        """
        # forward graph wiring
        forward_wiring = t.zeros(self.n_stages, self.N, dtype=t.long)
        forward_wiring[0] = self._get_reversed_bit_map(N=self.N, width=self.n_stages)
        for layer_i in range(1, self.n_stages):
            forward_wiring[layer_i] = self._get_butterfly_wiring(N=self.N, spread=2**(layer_i+1))
        # backward graph wiring
        backward_wiring = forward_wiring.clone()
        backward_wiring[0,::2] = t.arange(self.N//2)
        backward_wiring[0,1::2] = t.arange(self.N//2) + self.N//2

        return forward_wiring, backward_wiring.roll(shifts=-1, dims=0)

    def _get_butterfly_wiring(self, N, spread):
        """
        Computes the wiring between two butterfly stages.
        :param N: Size of FFT.
        :param spread: Size of independent wirings in this butterfly stage. E.g., for N=16, the first butterfly layer
                       has spread=4 and second stage has spread=8 and last stage has spread=16.
        :return: Tensor of shape (N,), representing the wiring to one butterfly stage.
        """
        assert (spread//2)*2 == spread
        assert (N//spread)*spread == N
        orig_range = t.arange(spread, device=self.dev)
        wiring_per_spread = t.arange(spread, device=self.dev)
        wiring_per_spread[1:spread//2:2] = orig_range[spread//2:-1:2]
        wiring_per_spread[spread//2:-1:2] = orig_range[1:spread//2:2]
        return spread * t.arange(N//spread, device=self.dev).repeat_interleave(spread) + wiring_per_spread.repeat(N//spread)

    def _get_reversed_bit_map(self, N, width):
        """
        For a bit width of <width>, reorder the decimals in range(0,N) according to the reversed binary representation.
        :param N: Size of range.
        :param width: Number of bits in binary representation.
        :return: Tensor of shape (N,) with rearranged decimal range(0,N).
        """
        reversed_bit_map = t.arange(N, dtype=t.int32)
        for n in range(N):  # TODO create LUT for large N?
            reversed_bit_map[n] = int('{:0{width}b}'.format(n, width=width)[::-1], 2)
        return reversed_bit_map

    def _get_buttefly_trafos(self):
        """
        For an FFT of size N, the transform matrices of size (log2(N),N/2,4,4) and the twiddle factors of all butterfly elements.
        Entry [i,j,:,:] corresponds to the linear trafo of the jth butterfly in stage i.
        The buttefly transforms 2 complex values (4 real values), thus is described as 4x4 matrix.
        :return: Tensor of shape (self.n_stages, N/2, 4, 4) and complex-valued tensor of shape (self.n_stages, N/2).
        """
        butterfly_graph = t.zeros(size=(self.n_stages, self.N//2, 4, 4), device=self.dev, dtype=self.prec)
        twiddle_per_stage = t.zeros(size=(self.n_stages, self.N//2), device=self.dev, dtype=self.cprec)
        for i in range(self.n_stages):
            w = t.exp(-1j*t.pi * t.arange(start=0, end=2**i, device=self.dev) / (2**i))
            twiddle_per_stage[i] = w.repeat(2 ** (self.n_stages - i - 1))

            butterfly_graph[i,:,0,0] = 1. * t.ones(self.N//2, dtype=t.float)
            butterfly_graph[i,:,0,2] =  w.real.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,0,3] = -w.imag.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,1,1] = 1. * t.ones(self.N//2, dtype=t.float)
            butterfly_graph[i,:,1,2] =  w.imag.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,1,3] =  w.real.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,2,0] = 1. * t.ones(self.N//2, dtype=t.float)
            butterfly_graph[i,:,2,2] = -w.real.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,2,3] =  w.imag.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,3,1] = 1. * t.ones(self.N//2, dtype=t.float)
            butterfly_graph[i,:,3,2] = -w.imag.repeat(2 ** (self.n_stages - i - 1))
            butterfly_graph[i,:,3,3] = -w.real.repeat(2 ** (self.n_stages - i - 1))

        return butterfly_graph, twiddle_per_stage
