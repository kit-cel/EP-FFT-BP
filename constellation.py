#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Code by Luca Schmid (luca.schmid@kit.edu)

import torch as t
import numpy as np

bpsk_mapping = t.tensor([1.0, -1.0], dtype=t.cfloat)

"""
Constellation class.
"""

class constellation:
    """
    Class which provides some functions, applied to an arbitrary complex constellation, 
    given in mapping.
    """

    def __init__(self, mapping, device):
        """
        :param mapping: t.Tensor which contains the constellation symbols, sorted according
            to their binary representation (MSB left).
        """
        assert len(mapping.shape) == 1 # mapping should be a 1-dim tensor
        self.mapping = mapping.to(device)
        
        self.M = t.numel(mapping) # Number of constellation symbols.
        self.m = np.log2(self.M).astype(int)
        assert self.m == np.log2(self.M) # Assert that log2(M) is integer
        self.mask = 2 ** t.arange(self.m - 1, -1, -1).to(device)

        self.sub_consts = t.stack([t.stack([t.arange(self.M).reshape(2**(i+1),-1)[::2].flatten(), t.arange(self.M).reshape(2**(i+1),-1)[1::2].flatten()]) for i in range(self.m)]).to(device)

        self.device = device

    def map(self, bits):
        """
        Maps a given bit_sequence to a sequence of constellation symbols.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1]/self.m == in_shape[-1]//self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return self.mapping[t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)]

    def bit2symbol_idx(self, bits):
        """
        Returns the symbol number (sorted as in self.mapping) for an incoming sequence of bits.
        This "symbol number" can be used for one-hot representation, for example.
        The length of the output sequence is len(bit_sequence) / m.
        The operation is applied to the last axis of bit_sequences.
        bit_sequence is allowed to have other dimensions (e.g. multiple sequences at once)
        as long as the last dimensions is the sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = bits.shape
        assert in_shape[-1]/self.m == in_shape[-1]//self.m
        # reshape and convert bits to decimal and use decimal number as index for mapping
        return t.sum(self.mask * bits.reshape(in_shape[:-1] + (-1, self.m)), -1)

    def demap(self, symbol_idxs):
        """
        Demaps a sequence of constellation symbols, given by their indices in self.mapping, to a sequence of bits.
        The length of the output sequence is len(symbols) * m.
        The operation is applied to the last axis of the input sequence.
        """
        # Assert that the length of the bit sequence is a multiple of m.
        in_shape = symbol_idxs.shape
        # reshape and convert symbol to bits and use decimal number as index for mapping
        return symbol_idxs.unsqueeze(-1).bitwise_and(self.mask).ne(0).view(symbol_idxs.shape[:-1]+(-1,)).float()

    def nearest_neighbor(self, rx_syms):
        """
        Accepts a sequence of (possibly equalized) complex symbols.
        Each sample is hard decided to the constellation symbol, which is nearest (Euclidean distance).
        The output are the idxs of the constellation symbols.
        """
        # Compute distances to all possible symbols.
        distance = t.abs(self.mapping - rx_syms[...,None])
        hard_dec_idx = t.argmin(distance, dim=-1)
        return hard_dec_idx
