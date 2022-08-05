import numpy as np
import os
import torch
import torch.nn.functional as F
from torch import nn
# from torchvision.models import alexnet
from torchsummary import summary
import timm
from efficientnet_pytorch import EfficientNet
import config as c
from freia_funcs import *

WEIGHT_DIR = './weights'
MODEL_DIR = './models'


def nf_head(input_dim=c.n_feat):
    nodes = list()
    nodes.append(InputNode(input_dim, name='input'))
    for k in range(c.n_coupling_blocks):
        # nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        nodes.append(Node([nodes[-1].out0], Norm, {}, name=F'norm_{k}'))
        nodes.append(Node([nodes[-1].out0], permute_layer, {'seed': k}, name=F'permute_{k}'))
        # nodes.append(Node([nodes[-1].out0], Invconv, {}, name=F'inconv_{k}'))
        nodes.append(Node([nodes[-1].out0], glow_coupling_layer,
                          {'clamp': c.clamp, 'F_class': F_conv,
                           'F_args': {'channels_hidden': c.fc_internal}},
                          name=F'fc_{k}'))
    nodes.append(OutputNode([nodes[-1].out0], name='output'))
    coder = ReversibleGraphNet(nodes)
    return coder


class ADwithGlow(nn.Module):
    def __init__(self):
        super(ADwithGlow, self).__init__()
        if c.extractor == 'wide_resnet':
            self.feature_extractor = timm.create_model(
                'wide_resnet50_2',
                pretrained=True,
                features_only=True,
                out_indices=[3],
            )
        else:
            self.feature_extractor = EfficientNet.from_pretrained('efficientnet-b5')
        self.nf = nf_head()

    def eff_ext(self, x, use_layer=36):  # 38可以尝试一下
        x = self.feature_extractor._swish(self.feature_extractor._bn0(self.feature_extractor._conv_stem(x)))
        # Blocks
        for idx, block in enumerate(self.feature_extractor._blocks):
            drop_connect_rate = self.feature_extractor._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.feature_extractor._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if idx == use_layer:
                return x
    def forward(self, x):
        if c.extractor == 'wide_resnet':
            feat_s = self.feature_extractor(x)
        else:
            feat_s = self.eff_ext(x)
        z = self.nf(feat_s)
        return z


def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    torch.save(model, os.path.join(MODEL_DIR, filename))


def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    model = torch.load(path)
    return model


def save_weights(model, filename):
    if not os.path.exists(WEIGHT_DIR):
        os.makedirs(WEIGHT_DIR)
    torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, filename))


def load_weights(model, filename):
    path = os.path.join(WEIGHT_DIR, filename)
    model.load_state_dict(torch.load(path))
    return model

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'models\\EfficientNet'
    x = torch.randn((1,3,256,256))
    model = ADwithGlow()
    # y = model(x)
    # jac = model.nf.jacobian(run_forward=False)
    # print(y.shape)
    # print(summary(model.feature_extractor,(3,256,256),device=c.device))
