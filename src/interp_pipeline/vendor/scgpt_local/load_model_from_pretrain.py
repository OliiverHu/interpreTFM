# This file is part of the vendored scGPT helpers of interpreTFM (GPL-3.0-or-later).
# The helpers contain code adapted from scGPT (https://github.com/bowang-lab/scGPT),
# Copyright (c) 2022 suber, licensed under the MIT License (see LICENSE.scGPT in this
# directory). Modified for interpreTFM.

# %%
import os
import torch

from nnsight import NNsight

from .model import scGPTModelClean, scGPTConfig
from .tokenizer import Tokenizer
from .util import load_pretrained


# %%
def create_clean_model_from_pretrain(model_dir, device="cpu"):
    model_file = os.path.join(model_dir, "best_model.pt")
    pretrained_params = torch.load(model_file, map_location=device)

    scgptcfg = scGPTConfig()
    scgptmodel = scGPTModelClean(scgptcfg)

    load_pretrained(scgptmodel, pretrained_params, verbose=True)
    scgptmodel.to(device)
    scgptmodel.eval()

    tokenizer = Tokenizer(os.path.join(model_dir, "vocab.json"), device=device)

    return NNsight(scgptmodel), tokenizer
