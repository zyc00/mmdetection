import torch.nn as nn
from ..builder import NECKS
from mmcv.runner import BaseModule, auto_fp16

@NECKS.register_module()
class PLANE(BaseModule):
    def __init__(self,
                in_channels,
                base_size, 
                init_cfg=None):
        super().__init__(init_cfg)

        self.dim_layers = nn.ModuleList([
            nn.Conv2d(in_channels, 256, 1),
            nn.LayerNorm((in_channels, base_size // 2 ** i))
        ] for i in range(4))

        self.proj_layers = nn.Module([
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.LayerNorm((256, base_size // 2 ** i))
        ] for i in range(4))

    @auto_fp16()
    def forward(self, inputs):
        outs = [proj_layer(dim_layer(input)) for proj_layer, dim_layer, input 
                            in zip(self.proj_layers, self.dim_layers, inputs)]
        return tuple(outs)