import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig
from esm.utils import encoding
from esm.utils.structure.affine3d import (
    Affine3D,
    build_affine3d_from_coordinates,
)
from esm.utils.constants import esm3 as C

from data import noise_utils
from model.vqvae import StructureTokenEncoder
from model.esm3_network import TimestepEmbedder
from model.inpainting_network import MaskedDiffusionLanguageModeling

from flowmatch.data import all_atom


def initialize_structure_encoder(args, pretrained_structure_encoder=None):
    print('Initialize structure encoder...')
    
    structure_encoder = StructureTokenEncoder(
        d_model=args.inpainting.structure_encoder_hidden,
        n_heads=args.inpainting.attn_n_heads,
        v_heads=args.inpainting.structure_encoder_v_heads,
        n_layers=args.inpainting.structure_encoder_layer,
        d_out=args.inpainting.structure_encoder_out,
        n_codes=C.VQVAE_CODEBOOK_SIZE,
        freeze_codebook=args.inpainting.freeze_codebook,
    )
    
    if pretrained_structure_encoder:
        structure_encoder.load_state_dict(pretrained_structure_encoder.state_dict())
        for param in structure_encoder.parameters():
            param.requires_grad = False

    return structure_encoder


def initialize_inpainting_module(args, esm3_model, noise_schedule=None, sigma_embedder=None, vqvae_model=None):
    print('Initialize inpainting module...')
    if noise_schedule is None:
        noise_schedule = noise_utils.get_noise(args.inpainting)
        
    if sigma_embedder is None:
        sigma_embedder = TimestepEmbedder(hidden_size=args.inpainting.sigma_embedder_hidden)
        
    inpainting_model = MaskedDiffusionLanguageModeling(
                                        args=args.inpainting,
                                        main_network=esm3_model,
                                        noise_schedule=noise_schedule,
                                        sigma_embedder=sigma_embedder,
                                    )
    
    if vqvae_model:
        print('Load pretrained vector-quantization model')
        inpainting_model.net._structure_encoder = vqvae_model
    
    return inpainting_model


class GENzyme(nn.Module):
    def __init__(self, model_conf, generative_model, inverse_folding_model, inpainting_model=None):
        super(GENzyme, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf

        self.generation_model = generative_model
        self.inversefold_model = inverse_folding_model
        self.inpainting_model = inpainting_model
        
                
    def frames_to_inversefold(self, frames):
        device = frames["amino_acid"].device
        batch_size, num_res, _ = frames["amino_acid"].shape

        _, atom_mask, _, atom_pos = all_atom.to_atom37(frames["rigids_tensor"])
        atom_pos = atom_pos[:, :, :4]
        atom_mask = atom_mask[:, :, 0]

        atom_pos = atom_pos.to(device)
        score = torch.zeros([batch_size, num_res]).to(device) + 100.0
        atom_mask = atom_mask.to(dtype=torch.float32).to(device)
        return atom_pos, score, atom_mask
    
    def quantize_frames(self, batch, frames, amino_acid=None):
        device = frames["amino_acid"].device
        batch_size, num_res = batch["token_mask"].shape
        _, _, _, pocket_pos = all_atom.to_atom37(frames["rigids_tensor"])
        pocket_pos = pocket_pos[..., :3, :].to(device)
        token_mask = batch["token_mask"].bool()
        pocket_mask = batch["pocket_mask"].bool()
        
        structure_tokens, sequence_tokens = [], []
        gt_structure_tokens = batch["structure_tokens"].detach().clone()
        nums_valid_res = batch["pocket_mask"].sum(dim=-1)
        for i in range(batch_size):
            n_valid_res = nums_valid_res[i]
            _, _structure_token = self.inpainting_model.net._structure_encoder.encode(
                pocket_pos[i][None, ...], residue_index=batch['residue_index'][i][None, ...]
            )
            
            # structure_token = torch.ones(1, num_res) * C.STRUCTURE_PAD_TOKEN
            # structure_token = structure_token.long().to(device)
            # structure_token[:, token_mask[i]] = C.STRUCTURE_MASK_TOKEN
            structure_token = gt_structure_tokens[i][None, ...]
            structure_token[:, pocket_mask[i]] = _structure_token[:, :n_valid_res]
            structure_tokens.append(structure_token)
            
            sequence_token = torch.ones(1, num_res) * C.SEQUENCE_PAD_TOKEN
            sequence_token = sequence_token.long().to(device)
            sequence_token[:, token_mask[i]] = C.SEQUENCE_MASK_TOKEN
            sequence_tokens.append(sequence_token)
            
        structure_tokens = torch.cat(structure_tokens, dim=0)
        sequence_tokens = torch.cat(sequence_tokens, dim=0)
        frames['structure_tokens'] = structure_tokens
        frames['sequence_tokens'] = sequence_tokens
        return frames


    def forward_frame(self, input_feats, use_context=False):
        frames = self.generation_model(input_feats)
        return frames

    def forward_inversefold(self, input_feats):
        pos, score, mask = input_feats
        X, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.inversefold_model._get_features(score, X=pos, mask=mask)
        aa_log_probs, aa_logits = self.inversefold_model(h_V, h_E, E_idx, batch_id, return_logit=True)
        return aa_log_probs, aa_logits
    
    def forward_inpaint_loss(self, batch, input_feats):
        loss = self.inpainting_model(batch, input_feats)
        return loss