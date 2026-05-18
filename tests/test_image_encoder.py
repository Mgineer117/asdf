import torch
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extractor.base.image_encoder import ImageEncoder, _jit_trace_encoder

chw_shape = (1, 210, 160)
device = "mps"

encoder = ImageEncoder(chw_shape, 256, device)
encoder = encoder.to(device)
encoder.eval()

print("Encoder device:", next(encoder.parameters()).device)

dummy = torch.zeros(1, *chw_shape, device=device)
print("Dummy device:", dummy.device)

traced = _jit_trace_encoder(encoder, chw_shape, device)

print("Traced!")
