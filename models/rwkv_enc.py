import torch
from models.rwkv_variants import ConfigurableRWKVSeg

def build_model(num_classes=1, input_size=256, pretrained=False, in_channels=3, **kwargs):
    return ConfigurableRWKVSeg(in_channels=in_channels, num_classes=num_classes, input_size=input_size, use_enc=True)
