import sys
import os
import torch
import torch.nn as nn
from transformers import AutoModel

print(f"Using Python: {sys.version}")
print(f"PyTorch version: {torch.version}")
print(f"CUDA available: {torch.cuda.is_available()}")

try:

    from base.base_model import BaseModel
    from model.video_transformer import SpaceTimeTransformer
    print("Base imports successful")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

from model.model import FrozenInTime
print("✅ FrozenInTime import successful")

# Test model initialization (without checkpoint first)
model = FrozenInTime(
    video_params={
        'model': 'SpaceTimeTransformer',
        'arch_config': 'base_patch16_224',
        'num_frames': 4,
        'pretrained': True,
        'time_init': 'zeros'
    },
    text_params={
        'model': 'distilbert-base-uncased',
        'pretrained': True,
        'input': 'text'
    },
    aggregation_params={
        'do_aggregation': True,
        'type': 'self-attention'
    },
    projection='minimal',
    load_checkpoint=None  # Test without checkpoint first
)
print("Model initialization successful (no checkpoint)")

# Test with checkpoint if available
if os.path.exists('/ivi/zfs/s0/original_homes/gmago/models/HierVL/hievl_sa.pth'):
    model_with_checkpoint = FrozenInTime(
        video_params={
            'model': 'SpaceTimeTransformer',
            'arch_config': 'base_patch16_224',
            'num_frames': 4,
            'pretrained': True,
            'time_init': 'zeros'
        },
        text_params={
            'model': 'distilbert-base-uncased',
            'pretrained': True,
            'input': 'text'
        },
        load_checkpoint='/ivi/zfs/s0/original_homes/gmago/models/HierVL/hievl_sa.pth'
    )
    print("Model with checkpoint loaded successfully")

