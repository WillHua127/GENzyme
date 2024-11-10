import os
from glob import glob
from pathlib import Path
from typing import Optional
from collections import OrderedDict

import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# import esm

from esm.models.vqvae import (
    StructureTokenDecoder,
    StructureTokenEncoder,
)
from esm.tokenization import StructureTokenizer
from esm.utils.decoding import decode_structure
from esm.utils import encoding, decoding, structure
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, GenerationConfig


SEQUENCE_VOCAB = [
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z",
    "O", ".", "-", "|",
    "<mask>",
]

RESTYPES = [
    "A", "R", "N", "D", "C",
    "Q", "E", "G", "H", "I",
    "L", "K", "M", "F", "P",
    "S", "T", "W", "Y", "V",
]

def cross_entropy(input, target, reduction='none', ignore_index=-100):
    # input = logits [B, L, V]
    # target = categories [B, L]
    logits_first = input.transpose(1, 2) # [B, V, L]
    return F.cross_entropy(logits_first, target, reduction=reduction, ignore_index=ignore_index) # [B, L]

def protstr_tokens_to_coords(
    structure_tokens: torch.Tensor,
    structure_decoder: StructureTokenDecoder,
    structure_tokenizer: StructureTokenizer,
    sequence: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    # https://github.com/evolutionaryscale/esm/blob/main/esm/utils/decoding.py#L139
    is_singleton = len(structure_tokens.size()) == 1
    if is_singleton:
        structure_tokens = structure_tokens.unsqueeze(0)
    else:
        raise ValueError(
            f"Only one structure can be decoded at a time, got structure tokens of shape {structure_tokens.size()}"
        )
    decoding._bos_eos_warn("Structure", structure_tokens[0], structure_tokenizer)

    decoder_output = structure_decoder.decode(structure_tokens)
    bb_coords: torch.Tensor = decoder_output["bb_pred"][
        0, 1:-1, ...
    ]  # Remove BOS and EOS tokens
    bb_coords = bb_coords.detach().cpu()

    if "plddt" in decoder_output:
        plddt = decoder_output["plddt"][0, 1:-1]
        plddt = plddt.detach().cpu()
    else:
        plddt = None

    if "ptm" in decoder_output:
        ptm = decoder_output["ptm"]
    else:
        ptm = None

    chain = ProteinChain.from_backbone_atom_coordinates(bb_coords, sequence=sequence)
    chain = chain.infer_oxygen()
    return torch.tensor(chain.atom37_positions), plddt, ptm
  
def prot_tensor_to_dict(prot):
    data_dict = {
        "sequence_tokens": prot.sequence,
        "structure_tokens": prot.structure,
        "ss8_tokens": prot.secondary_structure,
        "sasa_tokens": prot.sasa,
        "function_tokens": prot.function,
        "residue_annotation_tokens": prot.residue_annotations,
        "structure_coords": prot.coordinates,    
    }
    # Create batch dimension
    data_dict = {
        k: v.unsqueeze(0) for k,v in data_dict.items() if v is not None
    }
    return data_dict

@torch.no_grad()
def pdb_to_data(pdb_file):
    prot = ESMProtein.from_pdb(pdb_file)
    return prot
    #return protseq_to_data(sequence=prot.sequence, coordinates=prot.coordinates, **kwargs)
        

@torch.no_grad()
def protseq_to_data(
    sequence: str, 
    model: ESM3, 
    # device: torch.device = torch.device('cpu'), 
    coordinates: torch.Tensor | None = None,
    encode_only: bool = False,
    mask_ids: Optional[list] = None,
    filled_ids: Optional[list] = None,
    total_size: Optional[int] = None,
):
    # in: sequence
    # out: z, structure tokens, mask, ...
    if mask_ids is not None:
        sequence = list(sequence)
        for idx in mask_ids:
            assert 0 <= idx < len(sequence), f"Invalid mask index {idx} for sequence of length {len(sequence)}"
            sequence[idx] = '_'
            coordinates[idx] = float("Inf")
        sequence = ''.join(sequence)
    elif filled_ids is not None:
        # sanity check
        assert total_size is not None, "total_size must be provided when fill_ids is not None"
        assert all(0 <= idx < total_size for idx in fill_ids), f"Invalid fill index {fill_ids} for sequence of length {total_size}"
        _seq = ['_'] * total_size
        _coord = coordinates.new_ones(total_size, 37, 3) * float("Inf")
        for idx in filled_ids:
            _seq[idx] = sequence[idx]
            _coord[idx] = coordinates[idx]
        sequence = ''.join(_seq)
        coordinates = _coord

    prot = ESMProtein(sequence=sequence, coordinates=coordinates)
    gt_tokens = model.encode(prot)
    if encode_only:
        return {
            # [L, ]
            "sequence_tokens": gt_tokens.sequence,
            "structure_tokens": gt_tokens.structure, # None if no coordinates input, is tensor during training 
            # [L, ]
            "sequence": sequence, 
            "coordinates": coordinates,
        }

    protseq = ESMProtein(sequence=sequence)
    input_tokens = model.encode(protseq)

    kw_tokens = prot_tensor_to_dict(input_tokens)
    outs = model(**kw_tokens)
    data_dict = {
        # [L+2, *]
        "embeddings": outs.embeddings.squeeze(0),
        "structure_logits": outs.structure_logits.squeeze(0),
        "sequence_tokens": input_tokens.sequence,
        "sequence": sequence,   # [L, ]
        "coordinates": coordinates, # [L, 37, 3]  
        # label
        "structure_tokens": gt_tokens.structure, # None if no coordinates input, the label during training 
       
    }
    return data_dict   
