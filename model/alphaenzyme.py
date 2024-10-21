import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np


class AlphaEnzyme(nn.Module):
    def __init__(self, model_conf, generative_model, inverse_folding_model, inpainting_model=None):
        super(AlphaEnzyme, self).__init__()
        torch.set_default_dtype(torch.float32)
        self._model_conf = model_conf

        self.gen_model = generative_model
        self.inversefold_model = inverse_folding_model
        self.inpainting_model = inpainting_model

    def forward_frame(self, input_feats, use_context=False):
        frames = self.gen_model(input_feats)
        return frames

    def forward_inversefold(self, input_feats):
        pos, score, mask = input_feats
        X, score, h_V, h_E, E_idx, batch_id, mask_bw, mask_fw, decoding_order = self.inversefold_model._get_features(score, X=pos, mask=mask)
        aa_log_probs = self.inversefold_model(h_V, h_E, E_idx, batch_id)
        return aa_log_probs