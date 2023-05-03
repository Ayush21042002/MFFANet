from models.blocks import *
import torch
from torch import nn
import numpy as np


class MFFANet(nn.Module):
    """Deep residual network with spatial attention for face SR.
    # Arguments:
        - n_ch: base convolution channels
        - down_steps: how many times to downsample in the encoder
        - res_depth: depth of residual layers in the main body 
        - up_res_depth: depth of residual layers in each upsample block

    """
    def __init__(
        self,
        min_ch=32,
        max_ch=128,
        in_size=128,
        out_size=128,
        min_feat_size=16,
        res_depth=10,
        relu_type='leakyrelu',
        norm_type='bn',
        att_name='mffa3d',
        bottleneck_size=4,
    ):
        super(MFFANet, self).__init__()
        nrargs = {'norm_type': norm_type, 'relu_type': relu_type}

        ch_clip = lambda x: max(min_ch, min(x, max_ch))

        down_steps = int(np.log2(in_size // min_feat_size))
        up_steps = int(np.log2(out_size // min_feat_size))
        n_ch = ch_clip(max_ch // int(np.log2(in_size // min_feat_size) + 1))

        # ------------ define dehazing block --------------------
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(3, 3, 3, 1, 1, bias=True)

        self.e_conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.e_conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.e_conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.e_conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)

        # ------------ define encoder --------------------
        self.encoder = []
        self.encoder.append(ConvLayer(3, n_ch, 3, 1))
        hg_depth = int(np.log2(64 / bottleneck_size))
        for i in range(down_steps):
            cin, cout = ch_clip(n_ch), ch_clip(n_ch * 2)
            self.encoder.append(ResidualBlock(cin, cout, scale='down', hg_depth=hg_depth, att_name=att_name, **nrargs))

            n_ch = n_ch * 2
            hg_depth = hg_depth - 1
        # hg_depth = hg_depth + 1
        self.encoder = nn.Sequential(*self.encoder)

        # ------------ define residual layers --------------------
        self.res_layers = []
        for i in range(res_depth + 3 - down_steps):
            channels = ch_clip(n_ch)
            self.res_layers.append(ResidualBlock(channels, channels, hg_depth=hg_depth, att_name=att_name, **nrargs))
        self.res_layers = nn.Sequential(*self.res_layers)

        # ------------ define decoder --------------------
        self.decoder = []
        for i in range(up_steps):
            hg_depth = hg_depth + 1
            cin, cout = ch_clip(n_ch), ch_clip(n_ch // 2)
            self.decoder.append(ResidualBlock(cin, cout, scale='up', hg_depth=hg_depth, att_name=att_name, **nrargs))
            n_ch = n_ch // 2

        self.decoder = nn.Sequential(*self.decoder)
        self.out_conv = ConvLayer(ch_clip(n_ch), 3, 3, 1)
    
    def forward(self, input_img):
        
        x1 = self.relu(self.conv(input_img))
        
        x1 = self.relu(self.e_conv1(x1))
        x2 = self.relu(self.e_conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.e_conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.e_conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.e_conv5(concat3))

        clean_image = self.relu((x5 * input_img) - x5 + 1)

        out = self.encoder(clean_image)
        out = self.res_layers(out)
        out = self.decoder(out)
        out_img = self.out_conv(out)
        return out_img

