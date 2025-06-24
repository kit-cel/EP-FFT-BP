# Copyright 2024, 2025 Luca Schmid.
import torch as t

def llrs2hardZF_f(llrs, rx_f):
    assert llrs.shape == rx_f.shape
    tx_hard_f = t.sign(llrs) # Hard decision
    chanest_zf_f = rx_f / tx_hard_f
    return chanest_zf_f

def complex_cov2block_diag(cov):
    assert len(cov.shape) == 4
    assert cov.shape[2:] == (2,2)
    batch_size = cov.shape[0]
    n = cov.shape[1]
    block_cov_out = t.zeros(size=(batch_size,2*n,2*n), dtype=t.float)

    for batch_i in range(batch_size):
        block_cov_out[batch_i] = t.block_diag(*cov[batch_i])

    return block_cov_out

def interleaved_cov2cov(cov):
    assert len(cov.shape) == 3
    assert cov.shape[1] == cov.shape[2]
    batch_size = cov.shape[0]
    n = cov.shape[1]//2
    assert 2*n == cov.shape[1]
    cov_out = t.zeros(size=(batch_size,n,2,2), dtype=t.float)
    for ni in range(n):
        cov_out[:,ni,:,:] = cov[:,2*ni:2*(ni+1),2*ni:2*(ni+1)]
    return cov_out
