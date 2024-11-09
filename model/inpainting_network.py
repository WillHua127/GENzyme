import os
from typing import Any, Dict, Tuple, Optional, Union
from random import random
from copy import deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from tqdm import tqdm
from torchmetrics import MinMetric, MeanMetric
from esm.utils.constants import esm3 as C

from data import noise_utils
from data.esm_utils import cross_entropy

from model.esm3_network import TimestepEmbedder


def _sample_categorical(categorical_probs):
    gumbel_norm = (
        1e-10
        - (torch.rand_like(categorical_probs) + 1e-10).log())
    return (categorical_probs / gumbel_norm).argmax(dim=-1)


def _unsqueeze(x, reference):
    return x.view(
        * x.shape,
        * ((1,) * (len(reference.shape) - len(x.shape))))


@dataclass
class EncoderOutput:
    """Class for keeping track of an item in inventory."""
    last_hidden_state: torch.Tensor



class MaskedDiffusionLanguageModeling(nn.Module):
    def __init__(
        self,
        args,
        # modules
        main_network,
        noise_schedule: noise_utils.Noise = None,
        sigma_embedder: nn.Module = None,
        **kwargs,
    ):
        super(MaskedDiffusionLanguageModeling, self).__init__()
        if noise_schedule is None: 
            print("Using default noise schedule: CosineNoise(eps=1e-3)")
            noise_schedule = noise_utils.CosineNoise(eps=1e-3)
            
        if sigma_embedder is None: 
            print("Using default time embedder")
            sigma_embedder = TimestepEmbedder(hidden_size=args.sigma_embedder_hidden)
            
        # not change this
        T = 0
        sampling_eps = 1e-3
        # main flags
        time_conditioning = args.time_conditioning
        change_of_variables = args.change_of_variables
        importance_sampling = args.importance_sampling
        condition_dropout = args.condition_dropout
        condition_mask_rate = args.condition_mask_rate
        sequence_prediction = args.sequence_prediction
        noise_removal = args.noise_removal
        antithetic_sampling = args.antithetic_sampling
        structure_only = args.structure_only
        coupled_condition_mask = args.coupled_condition_mask
        
        
        self.net = main_network
        self.noise = noise_schedule    # nn.module
        self.sigma_embedder = sigma_embedder  # nn.module (B) -> (B, D)

        # flags
        self.antithetic_sampling = antithetic_sampling
        self.change_of_variables = change_of_variables
        self.importance_sampling = importance_sampling
        self.time_conditioning = time_conditioning
        self.T = T  # for discrete time markov chain
        self.sampling_eps = sampling_eps
        self.noise_removal = noise_removal
        self.structure_only = structure_only
        self.coupled_condition_mask = coupled_condition_mask
        
        # new features
        self.condition_dropout = condition_dropout
        self.condition_mask_rate = condition_mask_rate
        self.sequence_prediction = sequence_prediction
        
        # check
        if condition_dropout:
            assert 0.0 <= condition_dropout < 1.0, f"Invalid condition_dropout: {condition_dropout}"
        if condition_mask_rate:
            assert not coupled_condition_mask, f"coupled_condition_mask has to be False when condition_mask_rate is not 0.0"
            assert 0.0 <= condition_mask_rate < 1.0, f"Invalid condition_mask_rate: {condition_mask_rate}"
        if sequence_prediction:
            assert self.net.output_heads.sequence_head is not None, f"Sequence head not found in tbe network, but sequence_prediction is True."

        assert not (self.change_of_variables and self.importance_sampling)

        # constants
        self.vocab_size = 4101  # VQVAE_CODEBOOK_SIZE + 5 special tokens
        self.mask_index = C.STRUCTURE_MASK_TOKEN
        self.condition_mask_index = C.SEQUENCE_MASK_TOKEN
        self.neg_infinity = -1000000.0
            
            

    # for finetuning "esm3" model (BERT)
    def forward(self, batch, frames, training=True):
        labels = batch["structure_tokens"].detach().clone()
        x0 = frames["structure_tokens"].detach().clone()
        condition_seq = None     # (B, L)
        B, L = labels.shape
        
        if self.condition_dropout > 0 and training: # in face only training/Validation call this model_step() in this class
            if random() < self.condition_dropout:
                condition_seq = None

        if self.condition_mask_rate > 0 and condition_seq is not None and training:
            mask = (torch.rand_like(condition_seq, dtype=torch.float) < self.condition_mask_rate) & (condition_seq != C.SEQUENCE_PAD_TOKEN)
            condition_seq = torch.where(mask, C.SEQUENCE_MASK_TOKEN, condition_seq) # mutate valid tokens to [MASK](seq)
        
        # Padding (B, L)
        loss_mask = batch["token_mask"] * (labels != C.STRUCTURE_PAD_TOKEN)
        
        # diffusion masking, get xt and time
        t = self._sample_t(x0.shape[0], x0.device)
        if self.T > 0:  # round time step
            t = (t * self.T).to(torch.int) / self.T # t \in {1/T, 2/T, ..., 1}
            t += (1 / self.T)

        if self.change_of_variables:
            net_conditioning = t[:, None]
            f_T = torch.log1p(- torch.exp(- self.noise.sigma_max))
            f_0 = torch.log1p(- torch.exp(- self.noise.sigma_min))
            move_chance = torch.exp(f_0 + t * (f_T - f_0))
            move_chance = move_chance[:, None]
        else:
            sigma, dsigma = self.noise(t)
            net_conditioning = sigma[:, None]
            move_chance = 1 - torch.exp(-sigma[:, None])

        if self.structure_only: # no condition_seq
            condition_seq = None 

        xt, condition_seq = self.q_xt(x0, move_chance, condition_seq=condition_seq, non_moving_mask=batch.get("pocket_mask", None))
        # forward pass
        logits, seq_logits = self._model_wrapper(xt, condition_seq, net_conditioning)
                    
        if torch.isnan(logits).any():
            print("Model output logits", logits)
        
        # score parameterization, continuous time.
        log_p_theta = torch.gather(
            input=logits,
            dim=-1,
            index=x0[:, :, None],
        ).squeeze(-1)
        
        if self.change_of_variables or self.importance_sampling:
            loss = log_p_theta * torch.log1p(- torch.exp(- self.noise.sigma_min))
        else:
            # dsigma: (B, )
            # NLL loss
            loss = - log_p_theta * (dsigma / torch.expm1(sigma))[:, None]
        
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-9)   # fl
        loss_bd = {"nelbo": loss.detach().clone()}
        
        # auxilliary loss: always recover sequence
        if self.sequence_prediction:
            assert seq_logits is not None, f"Sequence logits not found in the forward pass."
            # (B, L, V)
            seq_reconstruction = cross_entropy(
                seq_logits, batch["sequence_tokens"], ignore_index=C.SEQUENCE_PAD_TOKEN
            )
            # note that here the loss mask is equivalent to non-padding mask
            # which different from "only count the masked tokens"
            # so it is readily applicable for auxilliary loss
            seq_reconstruction = (seq_reconstruction * loss_mask).sum() / (loss_mask.sum() + 1e-9)
            loss = loss + seq_reconstruction
            loss_bd["seq_nll"] = seq_reconstruction.detach().clone()

        return loss, loss_bd
    
    def _model_wrapper(self, xt, sequence_tokens=None, sigma=None, shield_special_tokens=False):
        # create time condition
        if sigma is not None:
            model_dtype = self.sigma_embedder.parameters().__next__().dtype
            sigma = self._process_sigma(sigma) # align with xt
            sigma = sigma.to(model_dtype)
            conditions = self.sigma_embedder(sigma)
            conditions = torch.tile(conditions[:, None, :], (1, xt.shape[1], 1))
        else:
            conditions = None   # no time conditioning (vanilla finetuning of BERT)
        
        _forward_output = self.net(
            structure_tokens=xt,
            sequence_tokens=sequence_tokens,
            auxiliary_embeddings=conditions,
            labels=None,
        )
        logits = _forward_output.structure_logits
        logits = self.logits_parameterization(logits=logits, xt=xt)
       
        if shield_special_tokens:
            for i in range(C.VQVAE_CODEBOOK_SIZE, C.VQVAE_CODEBOOK_SIZE + 5):
                logits[..., i] += self.neg_infinity

        if self.sequence_prediction:
            sequence_logits = _forward_output.sequence_logits
            return logits, sequence_logits    
    
        return logits, None

    def q_xt(self, x, move_chance, condition_seq=None, non_moving_mask=None):
        """Computes the noisy sample xt.
        
        Args:
            x: long with shape (B, L)
            move_chance: float with shape (B, )
            condition_seq: long with shape (B, L)
        """
        # fully random masking
        move_indices = torch.rand(*x.shape, device=x.device) < move_chance
        # if not None, mask the tokens other than the non-moving tokens
        # non_moving_mask: (B, L), 1 for non-moving tokens
        if non_moving_mask is not None:
            non_moving_mask = non_moving_mask.bool()
            move_indices = move_indices & (~non_moving_mask)

        xt = torch.where(move_indices, self.mask_index, x)
        if self.coupled_condition_mask and condition_seq is not None:
            condition_seq = torch.where(move_indices, self.condition_mask_index, condition_seq)     # mask the sequence tokens as well, coupled position
        return xt, condition_seq

    def _sample_prior(self, *batch_dims):
        return self.mask_index * torch.ones(*batch_dims, dtype=torch.int64)

    def _sample_t(self, n, device):
        _eps_t = torch.rand(n, device=device)
        if self.antithetic_sampling:
            offset = torch.arange(n, device=device) / n
            _eps_t = (_eps_t / n + offset) % 1
        t = (1 - self.sampling_eps) * _eps_t + self.sampling_eps
        if self.importance_sampling:
            return self.noise.importance_sampling_transformation(t)
        return t
    
    def logits_parameterization(self, logits, xt):
        logits[:, :, self.mask_index] += self.neg_infinity
        logits = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        unmasked_indices = (xt != self.mask_index)
        logits[unmasked_indices] = self.neg_infinity
        logits[unmasked_indices, xt[unmasked_indices]] = 0
        return logits   # (B, L, V+1)

    def _process_sigma(self, sigma):
        if sigma.ndim > 1:
            sigma = sigma.squeeze(-1)
        if not self.time_conditioning:
            sigma = torch.zeros_like(sigma)
        assert sigma.ndim == 1, sigma.shape
        return sigma

    @torch.no_grad()
    def ddpm_sample(self, sequence_tokens, num_steps=None, eps=1e-5, input_prior=None, sample_max_t=1.0, last_step_output_seq=False):
        """Generate samples from the model."""
        self.net.eval()
        self.noise.eval()
        self.sigma_embedder.eval()

        # Lightning auto-casting is not working in this method for some reason        
        if num_steps is None:
            num_steps = 1000
            
        if input_prior is None:
            x = self._sample_prior(*sequence_tokens.shape).to(sequence_tokens.device)
            assert sample_max_t == 1.0, f"sample_max_t has to be 1.0 when input_prior is None"
            
        else:
            # partially masked tokens
            # round trip diffusion
            x = input_prior.to(sequence_tokens.device)
            assert x.shape == sequence_tokens.shape, f"Invalid input_prior shape: {x.shape} v.s. (seq) {sequence_tokens.shape}"

        timesteps = torch.linspace(
            sample_max_t, eps, num_steps + 1, device=x.device
        )
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        for i in tqdm(range(num_steps), desc="DDPM Sampling ..."):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=x.device)
            x = self._ddpm_update(x, t, sequence_tokens=sequence_tokens, dt=dt)

        print(self.noise_removal)
        if self.noise_removal:
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=x.device)
            sigma_t = self.noise(t)[0]
            x, seq_log = self._model_wrapper(x, sequence_tokens, sigma_t)
            x = x.argmax(dim=-1)

        return x
    
    
    @torch.no_grad()
    def ddpm_sample_trajectory(self, sequence_tokens, num_steps=None, eps=1e-5, input_prior=None, sample_max_t=1.0, last_step_output_seq=False):
        """Generate samples from the model."""
        self.net.eval()
        self.noise.eval()
        self.sigma_embedder.eval()

        # Lightning auto-casting is not working in this method for some reason        
        if num_steps is None:
            num_steps = 1000
            
        if input_prior is None:
            x = self._sample_prior(*sequence_tokens.shape).to(sequence_tokens.device)
            assert sample_max_t == 1.0, f"sample_max_t has to be 1.0 when input_prior is None"
            
        else:
            # partially masked tokens
            # round trip diffusion
            x = input_prior.to(sequence_tokens.device)
            assert x.shape == sequence_tokens.shape, f"Invalid input_prior shape: {x.shape} v.s. (seq) {sequence_tokens.shape}"

        timesteps = torch.linspace(
            sample_max_t, eps, num_steps + 1, device=x.device
        )
        dt = (1 - eps) / num_steps
        p_x0_cache = None

        trajectory = []
        for i in tqdm(range(num_steps), desc="DDPM Sampling ..."):
            t = timesteps[i] * torch.ones(
                x.shape[0], 1, device=x.device)
            x = self._ddpm_update(x, t, sequence_tokens=sequence_tokens, dt=dt)

            sigma_t = self.noise(t)[0]
            _x, seq_log = self._model_wrapper(x, sequence_tokens, sigma_t)
            _x = _x.argmax(dim=-1)
            trajectory.append(_x)

        return trajectory
    
    def _ddpm_update(self, x, t, sequence_tokens, dt):
        sigma_t, _ = self.noise(t)
        sigma_s, _ = self.noise(t - dt)
        if sigma_t.ndim > 1:
            sigma_t = sigma_t.squeeze(-1)
        if sigma_s.ndim > 1:
            sigma_s = sigma_s.squeeze(-1)
        assert sigma_t.ndim == 1, sigma_t.shape
        assert sigma_s.ndim == 1, sigma_s.shape
        move_chance_t = 1 - torch.exp(-sigma_t)
        move_chance_s = 1 - torch.exp(-sigma_s)
        move_chance_t = move_chance_t[:, None, None]
        move_chance_s = move_chance_s[:, None, None]
        # conditional sampling
        log_p_x0, _ = self._model_wrapper(x, sequence_tokens, sigma_t)
        assert move_chance_t.ndim == log_p_x0.ndim
        # Technically, this isn't q_xs since there's a division
        # term that is missing. This division term doesn't affect
        # the samples.
        q_xs = log_p_x0.exp() * (move_chance_t - move_chance_s)
        q_xs[:, :, self.mask_index] = move_chance_s[:, :, 0]    
        _x = _sample_categorical(q_xs)

        copy_flag = (x != self.mask_index).to(x.dtype)
        return copy_flag * x + (1 - copy_flag) * _x