"""Conditional Generative models defined for structure tokens used in ESM3.
"""
import os
import math
import torch 
from torch import nn
from torch import Tensor
from copy import deepcopy

from esm.utils.constants import esm3 as C
from esm.models.esm3 import ESM3, ESMOutput, EncodeInputs, OutputHeads
from esm.layers.transformer_stack import TransformerStack
from esm.layers.regression_head import RegressionHead
from esm.utils.structure.affine3d import (
    build_affine3d_from_coordinates,
)
from esm.tokenization import get_model_tokenizers
from esm.utils.constants.models import (
    ESM3_OPEN_SMALL,
)
from esm.pretrained import (
    load_local_model, 
    ESM3_structure_encoder_v0, 
    ESM3_structure_decoder_v0, 
    ESM3_function_decoder_v0,
)


ESM3_D_MODEL = 1536
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# helper Module that adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(
        self,
        emb_size: int,
        dropout: float,
        maxlen: int = 10000
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
    
    
class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size=ESM3_D_MODEL, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True))
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            - math.log(max_period)
            * torch.arange(start=0, end=half, dtype=t.dtype)
            / half).to(device=t.device, dtype=t.dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
            [embedding,
            torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def _shift_right(input_ids, pad_token_id):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = pad_token_id
    return shifted_input_ids

def _generate_square_subsequent_mask(
    sz: int,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    """Generate a square causal mask for the sequence.

    The masked positions are filled with float('-inf'). Unmasked positions are filled with float(0.0).
    """
    if device is None:
        device = torch.device('cpu')
    if dtype is None:
        dtype = torch.float32
    return torch.triu(
        torch.full((sz, sz), float('-inf'), dtype=dtype, device=device),
        diagonal=1,
    )


class StructureOutputHeads(nn.Module):
    def __init__(self, d_model: int, n_structure_heads: int = 4096, n_sequence_heads: int = 0):
        super().__init__()
        self.structure_head = RegressionHead(d_model, n_structure_heads)
        if n_sequence_heads:
            self.sequence_head = RegressionHead(d_model, n_sequence_heads)
        else:
            self.sequence_head = None

    def forward(self, x: torch.Tensor, embed: torch.Tensor) -> ESMOutput:
        structure_logits = self.structure_head(x)
        dummy_tensor = torch.zeros_like(structure_logits)
        sequence_logits = self.sequence_head(x) \
            if self.sequence_head is not None else dummy_tensor
        return ESMOutput(
            sequence_logits=sequence_logits,
            structure_logits=structure_logits,
            secondary_structure_logits=dummy_tensor,
            sasa_logits=dummy_tensor,
            function_logits=dummy_tensor,
            residue_logits=dummy_tensor,
            embeddings=embed,
        )

    
class CustomizedESM3(ESM3):
    def __init__(
        self, 
        args,
        **kwargs,
    ):
        # config of ESM3_OPEN_SMALL
        super(ESM3, self).__init__()
        d_model = args.sigma_embedder_hidden
        n_heads = args.attn_n_heads
        v_heads = args.attn_v_heads
        n_layers = args.n_layers
        pretrained = args.pretrained
        n_structure_heads = args.n_structure_heads
        n_sequence_heads = args.n_sequence_heads
        
        self.encoder = EncodeInputs(d_model=d_model)
        
        self.transformer = TransformerStack(
            d_model,
            n_heads,
            v_heads,
            n_layers,
            mask_and_zero_frameless=True,
        )
        self.output_heads = OutputHeads(d_model)

        self.structure_encoder_fn = ESM3_structure_encoder_v0
        self.structure_decoder_fn = ESM3_structure_decoder_v0
        self.function_decoder_fn = ESM3_function_decoder_v0

        self._structure_encoder = None
        self._structure_decoder = None
        self._function_decoder = None

        self.tokenizers = get_model_tokenizers(ESM3_OPEN_SMALL)
        
        if pretrained:
            print("Load pretrained esm3 model...")
            model = load_local_model(ESM3_OPEN_SMALL) # if eval mode, will overwrite later
            self.load_state_dict(model.state_dict())
        
        if n_structure_heads != C.VQVAE_CODEBOOK_SIZE:
            # replace used head
            print(f">>> [CustomizedESM3] Replace output_heads() with n_structure_heads={n_structure_heads}, n_sequence_heads={n_sequence_heads}")
            self.output_heads = StructureOutputHeads(d_model, n_structure_heads, n_sequence_heads)
            
        self.d_model = d_model
        self.train()

    def forward(
        self, 
        structure_tokens: torch.Tensor,
        labels: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sequence_tokens: torch.Tensor | None = None,    # for decoder
        *,
        encoder_embeddings: torch.Tensor | None = None,
        ss8_tokens: torch.Tensor | None = None,
        sasa_tokens: torch.Tensor | None = None,
        function_tokens: torch.Tensor | None = None,
        residue_annotation_tokens: torch.Tensor | None = None,
        average_plddt: torch.Tensor | None = None,
        per_res_plddt: torch.Tensor | None = None,
        structure_coords: torch.Tensor | None = None,
        chain_id: torch.Tensor | None = None,
        sequence_id: torch.Tensor | None = None,
        auxiliary_embeddings: torch.Tensor | None = None,
    ):
        """Forward pass and get loss. *Tailored for conformation generation task.*
        """

        try:
            L, device = next(
                (x.shape[1], x.device)
                for x in [
                    sequence_tokens,
                    structure_tokens,
                    ss8_tokens,
                    sasa_tokens,
                    structure_coords,
                    function_tokens,
                    residue_annotation_tokens,
                ]
                if x is not None
            )
        except StopIteration:
            raise ValueError("At least one of the inputs must be non-None")

        t = self.tokenizers
        defaults = lambda x, tok: (
            torch.full((1, L), tok, dtype=torch.long, device=device) if x is None else x
        )
        sequence_tokens = defaults(sequence_tokens, t.sequence.mask_token_id)
        ss8_tokens = defaults(ss8_tokens, C.SS8_PAD_TOKEN)
        sasa_tokens = defaults(sasa_tokens, C.SASA_PAD_TOKEN)
        chain_id = defaults(chain_id, 0)
        
        # non long dtype
        average_plddt = defaults(average_plddt, 1).float()
        per_res_plddt = defaults(per_res_plddt, 0).float()

        if residue_annotation_tokens is None:
            residue_annotation_tokens = torch.full(
                (1, L, 16), C.RESIDUE_PAD_TOKEN, dtype=torch.long, device=device
            )

        if function_tokens is None:
            function_tokens = torch.full(
                (1, L, 8), C.INTERPRO_PAD_TOKEN, dtype=torch.long, device=device
            )

        if structure_coords is None:
            structure_coords = torch.full(
                (1, L, 3, 3), float("nan"), dtype=torch.float, device=device
            )

        structure_coords = structure_coords[
            ..., :3, :
        ]  # In case we pass in an atom14 or atom37 repr
        affine, affine_mask = build_affine3d_from_coordinates(structure_coords)

        structure_tokens = defaults(structure_tokens, C.STRUCTURE_MASK_TOKEN)
        assert structure_tokens is not None
        structure_tokens = (
            structure_tokens.masked_fill(structure_tokens == -1, C.STRUCTURE_MASK_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_BOS_TOKEN, C.STRUCTURE_BOS_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_PAD_TOKEN, C.STRUCTURE_PAD_TOKEN)
            .masked_fill(sequence_tokens == C.SEQUENCE_EOS_TOKEN, C.STRUCTURE_EOS_TOKEN)
            .masked_fill(
                sequence_tokens == C.SEQUENCE_CHAINBREAK_TOKEN,
                C.STRUCTURE_CHAINBREAK_TOKEN,
            )
        )
        x = self.encoder(
            sequence_tokens,
            structure_tokens,
            average_plddt,
            per_res_plddt,
            ss8_tokens,
            sasa_tokens,
            function_tokens,
            residue_annotation_tokens,
        )
        if auxiliary_embeddings is not None:
            x = x + auxiliary_embeddings
        
        x, embedding = self.transformer(x, sequence_id, affine, affine_mask, chain_id)
        forward_output = self.output_heads(x, embedding) # ESMOutput
        
        if labels is not None:
            assert mask is not None, "mask must be provided when labels is not None"
            return_dict = {
                "decoder_embeddings": forward_output.embeddings,
                "structure_logits": forward_output.structure_logits,
                "sequence_logits": forward_output.sequence_logits,
            }
            return return_dict
        
        return forward_output   # inference