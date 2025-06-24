# Uncertainty Propagation in the Fast Fourier Transform
### A framework for uncertainty-aware Bayesian inference in probabilistic systems operating across both time and frequency domain

We address the problem of uncertainty propagation in the discrete Fourier transform (DFT) by modeling the fast Fourier transform (FFT) as a factor graph. Building on this representation, we propose an efficient framework for approximate Bayesian inference using belief propagation (BP) and expectation propagation, extending its applicability beyond Gaussian assumptions. By leveraging an appropriate BP message representation and a suitable schedule, our method achieves stable convergence with accurate mean and variance estimates. Numerical experiments in representative scenarios from communications demonstrate the practical potential of the proposed framework for uncertainty-aware inference in probabilistic systems operating across both time and frequency domain.

---

This repo contains the source code for the [EP-FFT framework](https://arxiv.org/pdf/2504.10136) [1]. It provides:
* Generic, modular and well-documented code of the EP-FFT framework. The notations and comments are aligned with [1].
* 2 experiments to showcase the application of EP-FFT and to reproduce the numerical results in [1]
  * Symbol detection in a digital linear inter-symbol interference channel
  * Estimation of a channel impulse response with multiple dominant reflectors leading to a sparse power-delay profile, e.g., in the context of a multistatic joint communication and sensing scenario in the context of an OFDM transmission
 
If you're interested in contributing, reporting issues, asking questions, or collaborating, please don’t hesitate to get in touch.

[1] L. Schmid, C. Muth, L. Schmalen, ``Uncertainty Propagation in the Fast Fourier Transform'', accepted for publication at the _International Workshop on Signal Processing and Artificial Intelligence in Wireless Communications (SPAWC)_ , Surrey, UK, July 2025, preprint available at https://arxiv.org/pdf/2504.10136

---

This work has received funding in part from the European Research Council (ERC) under the European Union’s Horizon 2020 research and innovation programme (grant agreement No. 101001899) and in part from the German Federal Ministry of Research, Technology and Space (BMFTR) within the project Open6GHub (grant agreement 16KISK010).
