import collections.abc
import math
import torch
import torchvision
import warnings
from distutils.version import LooseVersion
from itertools import repeat
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm

from basicsr.ops.dcn import ModulatedDeformConvPack, modulated_deform_conv
from basicsr.utils import get_root_logger


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


class CCALayer(nn.Module):
    def __init__(self, channel, reduction=2):
        super(CCALayer, self).__init__()

        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, padding=0,
                                 stride=1,
                                 groups=1, bias=True)
        self.sig = nn.Sigmoid()
        #     nn.Sequential(
        #     nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        y = self.sig(self.conv_du(self.contrast(x) + self.avg_pool(x)))
        return x * y


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        # Simplified contrast-aware Channel Attention
        self.sca = CCALayer(channel=dw_channel, reduction=2)
        #     nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # SimpleGate
        self.sg = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=c,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=c, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x)
        x = self.conv3(x)
        # x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.sg(x)
        # x = self.dropout2(x)
        return y + x * self.gamma


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class DWResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, in_channel=64, num_feat=64, res_scale=1, pytorch_init=False):
        super(DWResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv0 = nn.Conv2d(in_channel, num_feat, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1, groups=num_feat, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1, groups=num_feat, bias=True)
        self.conv3 = nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, stride=1, groups=num_feat, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.esa = ESA(num_feat, nn.Conv2d)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2, self.conv3], 0.1)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.relu(self.conv3(self.relu(self.conv2(self.relu(self.conv1(x1))))))
        out = x1 + out * self.res_scale
        out = self.esa(out)
        # out = self.conv2(self.relu(self.conv1(x)))
        # return identity + out * self.res_scale
        return out


class ESA(nn.Module):
    def __init__(self, n_feats, conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        c3 = self.conv_f(self.relu(self.conv_max(v_max)))
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        c4 = self.conv4(c3 + c1_)
        m = self.sigmoid(c4)

        return x * m


class RFDB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RFDB, self).__init__()
        self.dc = self.distilled_channels = out_channels // 2
        self.rc = self.remaining_channels = out_channels
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.c1_d = nn.Conv2d(out_channels, self.dc, kernel_size=1, padding=0, stride=1, bias=True)
        self.c1_r = DWResidualBlockLayer(self.remaining_channels, self.rc, 1)
        self.c2_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0, stride=1, bias=True)
        self.c2_r = DWResidualBlockLayer(self.remaining_channels, self.rc, 1)
        self.c3_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0, stride=1, bias=True)
        self.c3_r = DWResidualBlockLayer(self.remaining_channels, self.rc, 1)
        self.c4 = DWResidualBlockLayer(self.remaining_channels, self.rc, 1)
        self.c4_d = nn.Conv2d(self.remaining_channels, self.dc, kernel_size=1, padding=0, stride=1, bias=True)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.c5 = nn.Conv2d(self.dc * 4, out_channels, kernel_size=1, padding=0, stride=1, bias=True)
        self.esa = ESA(out_channels, nn.Conv2d)

    def forward(self, input):
        input1 = self.pre(input)
        distilled_c1 = self.act(self.c1_d(input1))
        # print(distilled_c1.shape)
        r_c1 = (self.c1_r(input1))
        r_c1 = self.act(r_c1 + input1)
        # print(r_c1.shape)

        distilled_c2 = self.act(self.c2_d(r_c1))
        # print(distilled_c2)
        r_c2 = (self.c2_r(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        # print(distilled_c3)
        r_c3 = (self.c3_r(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4_d(self.act(self.c4(r_c3))))
        # print(r_c4.shape)

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.esa(self.c5(out))

        return out_fused


class DWResidualBlockLayer(nn.Module):

    def __init__(self, num_in_ch=3, num_out_ch=64, num_block=15):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_in_ch, num_out_ch, kernel_size=1, padding=0, stride=1, bias=True),
            make_layer(DWResidualBlockNoBN, num_block, num_feat=num_out_ch))

    def forward(self, fea):
        return self.main(fea)


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


class SeqConv3x3(nn.Module):

    def __init__(self, seq_type, inp_planes, out_planes, depth_multiplier=1):
        super(SeqConv3x3, self).__init__()
        self.seq_type = seq_type
        self.inp_planes = inp_planes
        self.out_planes = out_planes

        if self.seq_type == 'conv1x1-conv3x3':
            self.mid_planes = int(out_planes * depth_multiplier)
            conv0 = torch.nn.Conv2d(self.inp_planes, self.mid_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            conv1 = torch.nn.Conv2d(self.mid_planes, self.out_planes, kernel_size=3)
            self.k1 = conv1.weight
            self.b1 = conv1.bias

        elif self.seq_type == 'conv1x1-sobelx':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale and bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(scale)
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(bias)
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 1, 0] = 2.0
                self.mask[i, 0, 2, 0] = 1.0
                self.mask[i, 0, 0, 2] = -1.0
                self.mask[i, 0, 1, 2] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.seq_type == 'conv1x1-sobely':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale and bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 0] = 1.0
                self.mask[i, 0, 0, 1] = 2.0
                self.mask[i, 0, 0, 2] = 1.0
                self.mask[i, 0, 2, 0] = -1.0
                self.mask[i, 0, 2, 1] = -2.0
                self.mask[i, 0, 2, 2] = -1.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)

        elif self.seq_type == 'conv1x1-laplacian':
            conv0 = torch.nn.Conv2d(self.inp_planes, self.out_planes, kernel_size=1, padding=0)
            self.k0 = conv0.weight
            self.b0 = conv0.bias
            # init scale and bias
            scale = torch.randn(size=(self.out_planes, 1, 1, 1)) * 1e-3
            self.scale = nn.Parameter(torch.FloatTensor(scale))
            bias = torch.randn(self.out_planes) * 1e-3
            bias = torch.reshape(bias, (self.out_planes, ))
            self.bias = nn.Parameter(torch.FloatTensor(bias))
            # init mask
            self.mask = torch.zeros((self.out_planes, 1, 3, 3), dtype=torch.float32)
            for i in range(self.out_planes):
                self.mask[i, 0, 0, 1] = 1.0
                self.mask[i, 0, 1, 0] = 1.0
                self.mask[i, 0, 1, 2] = 1.0
                self.mask[i, 0, 2, 1] = 1.0
                self.mask[i, 0, 1, 1] = -4.0
            self.mask = nn.Parameter(data=self.mask, requires_grad=False)
        else:
            raise ValueError('The type of seqconv is not supported!')

    def forward(self, x):
        if self.seq_type == 'conv1x1-conv3x3':
            # conv-1x1
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.k1, bias=self.b1, stride=1)
        else:
            y0 = F.conv2d(input=x, weight=self.k0, bias=self.b0, stride=1)
            # explicitly padding with bias
            y0 = F.pad(y0, (1, 1, 1, 1), 'constant', 0)
            b0_pad = self.b0.view(1, -1, 1, 1)
            y0[:, :, 0:1, :] = b0_pad
            y0[:, :, -1:, :] = b0_pad
            y0[:, :, :, 0:1] = b0_pad
            y0[:, :, :, -1:] = b0_pad
            # conv-3x3
            y1 = F.conv2d(input=y0, weight=self.scale * self.mask, bias=self.bias, stride=1, groups=self.out_planes)
        return y1

    def rep_params(self):
        device = self.k0.get_device()
        if device < 0:
            device = None

        if self.seq_type == 'conv1x1-conv3x3':
            # re-param conv kernel
            rep_weight = F.conv2d(input=self.k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            rep_bias = torch.ones(1, self.mid_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=self.k1).view(-1, ) + self.b1
        else:
            tmp = self.scale * self.mask
            k1 = torch.zeros((self.out_planes, self.out_planes, 3, 3), device=device)
            for i in range(self.out_planes):
                k1[i, i, :, :] = tmp[i, 0, :, :]
            b1 = self.bias
            # re-param conv kernel
            rep_weight = F.conv2d(input=k1, weight=self.k0.permute(1, 0, 2, 3))
            # re-param conv bias
            rep_bias = torch.ones(1, self.out_planes, 3, 3, device=device) * self.b0.view(1, -1, 1, 1)
            rep_bias = F.conv2d(input=rep_bias, weight=k1).view(-1, ) + b1
        return rep_weight, rep_bias


class ECB(nn.Module):

    def __init__(self, inp_planes, out_planes, depth_multiplier, act_type='relu', with_idt=False, pytorch_init=False):
        super(ECB, self).__init__()

        self.depth_multiplier = depth_multiplier
        self.inp_planes = inp_planes
        self.out_planes = out_planes
        self.act_type = act_type

        if with_idt and (self.inp_planes == self.out_planes):
            self.with_idt = True
        else:
            self.with_idt = False

        self.conv1x1_3x3 = SeqConv3x3('conv1x1-conv3x3', self.inp_planes, self.out_planes, self.depth_multiplier)
        self.conv1x1_sbx = SeqConv3x3('conv1x1-sobelx', self.inp_planes, self.out_planes)
        self.conv1x1_sby = SeqConv3x3('conv1x1-sobely', self.inp_planes, self.out_planes)
        self.conv1x1_lpl = SeqConv3x3('conv1x1-laplacian', self.inp_planes, self.out_planes)

        if self.act_type == 'prelu':
            self.act = nn.PReLU(num_parameters=self.out_planes)
        elif self.act_type == 'lrelu':
            self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'rrelu':
            self.act = nn.RReLU(lower=-0.05, upper=0.05)
        elif self.act_type == 'softplus':
            self.act = nn.Softplus()
        elif self.act_type == 'linear':
            pass
        else:
            raise ValueError('The type of activation if not support!')

    def forward(self, x):
        if self.training:
            y = self.conv1x1_3x3(x) + self.conv1x1_sbx(x) + self.conv1x1_sby(x) + self.conv1x1_lpl(x)
            if self.with_idt:
                y += x
        else:
            rep_weight, rep_bias = self.rep_params()
            y = F.conv2d(input=x, weight=rep_weight, bias=rep_bias, stride=1, padding=1)
        if self.act_type != 'linear':
            y = self.act(y)
        return y

    def rep_params(self):
        weight1, bias1 = self.conv1x1_3x3.rep_params()
        weight2, bias2 = self.conv1x1_sbx.rep_params()
        weight3, bias3 = self.conv1x1_sby.rep_params()
        weight4, bias4 = self.conv1x1_lpl.rep_params()
        rep_weight, rep_bias = (weight1 + weight2 + weight3 + weight4), (bias1 + bias2 + bias3 + bias4)

        if self.with_idt:
            device = rep_weight.get_device()
            if device < 0:
                device = None
            weight_idt = torch.zeros(self.out_planes, self.out_planes, 3, 3, device=device)
            for i in range(self.out_planes):
                weight_idt[i, i, 1, 1] = 1.0
            bias_idt = 0.0
            rep_weight, rep_bias = rep_weight + weight_idt, rep_bias + bias_idt
        return rep_weight, rep_bias


class ECBGroup(nn.Module):

    def __init__(self, in_channel=64, num_feat=64, res_scale=1, num_block=3):
        super(ECBGroup, self).__init__()
        self.res_scale = res_scale
        self.conv0 = nn.Conv2d(in_channel, num_feat, kernel_size=1, padding=0, stride=1, bias=True)
        self.main = make_layer(ECB, num_block, inp_planes=num_feat, out_planes=num_feat, depth_multiplier=2)
        self.esa = ESA(num_feat, nn.Conv2d)

    def forward(self, x):
        x1 = self.conv0(x)
        out = self.main(x1)
        out = x1 + out * self.res_scale
        out = self.esa(out)
        # out = self.conv2(self.relu(self.conv1(x)))
        # return identity + out * self.res_scale
        return out


# class ECBCD(nn.Module):
#     # edge-oriented conv block with channel distillation
#     def __init__(self, in_channels, out_channels):
#         super(ECBCD, self).__init__()
#         self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
#         self.ecb1 = ECB(inp_planes=out_channels, out_planes=out_channels // 2, depth_multiplier=2.0, act_type='relu',
#                         with_idt=False)
#         self.cd1 = nn.Conv2d(out_channels, out_channels // 4, kernel_size=1, padding=0, stride=1)
#         self.ecb2 = ECB(inp_planes=out_channels // 2, out_planes=out_channels // 4, depth_multiplier=2.0, act_type='relu',
#                         with_idt=False)
#         self.cd2 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1, padding=0, stride=1)
#         self.ecb3 = ECB(inp_planes=out_channels // 4, out_planes=out_channels // 4, depth_multiplier=2.0, act_type='relu',
#                         with_idt=False)
#         self.cd3 = nn.Conv2d(out_channels // 4, out_channels // 4, kernel_size=1, padding=0, stride=1)
#         self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#         self.esa = ESA(out_channels, nn.Conv2d)

#     def forward(self, x):
#         x = self.pre(x)
#         x1 = self.ecb1(x)
#         x1_d = self.act(self.cd1(x))
#         x2 = self.ecb2(x1)
#         x2_d = self.act(self.cd2(x1))
#         x3 = self.ecb3(x2)
#         x3_d = self.act(self.cd3(x2))
#         out = torch.cat([x1_d, x2_d, x3_d, x3], dim=1)
#         out = self.esa(out)
#         return out

class ECBCD(nn.Module):
    # edge-oriented conv block with channel distillation
    def __init__(self, in_channels, out_channels):
        super(ECBCD, self).__init__()
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.cd0 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, padding=0, stride=1)
        self.ecb1 = ECB(inp_planes=out_channels // 2, out_planes=out_channels // 2, depth_multiplier=2.0, act_type='relu',
                        with_idt=False)

        self.cd1 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1, padding=0, stride=1)

        self.ecb2 = ECB(inp_planes=out_channels // 2, out_planes=out_channels // 2, depth_multiplier=2.0, act_type='relu',
                        with_idt=False)

        self.cd2 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1, padding=0, stride=1)

        self.ecb3 = ECB(inp_planes=out_channels, out_planes=out_channels, depth_multiplier=2.0, act_type='relu',
                        with_idt=False)

        self.act = nn.ReLU(inplace=True)
        self.esa = ESA(out_channels, nn.Conv2d)

    def forward(self, fea):
        fea = self.pre(fea)
        x = self.cd0(fea)
        x1 = self.ecb1(x)
        x1_d = self.act(self.cd1(x1))
        x2 = self.ecb2(x1)
        x2_d = self.act(self.cd2(x2))
        out = torch.cat([x1_d, x2_d, x2], dim=1)
        out += fea
        out = self.ecb3(out)
        out = self.esa(out)
        return out


class ConvCD(nn.Module):
    # edge-oriented conv block with channel distillation
    def __init__(self, in_channels, out_channels):
        super(ConvCD, self).__init__()
        self.pre = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, stride=1)
        self.cd0 = nn.Conv2d(out_channels, out_channels // 2, kernel_size=1, padding=0, stride=1)
        self.ecb1 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, padding=0, stride=1)

        self.cd1 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1, padding=0, stride=1)

        self.ecb2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=1, padding=0, stride=1)

        self.cd2 = nn.Conv2d(out_channels // 2, out_channels // 4, kernel_size=1, padding=0, stride=1)

        self.ecb3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0, stride=1)

        self.act = nn.ReLU(inplace=True)
        self.esa = ESA(out_channels, nn.Conv2d)

    def forward(self, fea):
        fea = self.pre(fea)
        x = self.cd0(fea)
        x1 = self.ecb1(x)
        x1_d = self.act(self.cd1(x1))
        x2 = self.ecb2(x1)
        x2_d = self.act(self.cd2(x2))
        out = torch.cat([x1_d, x2_d, x2], dim=1)
        out += fea
        out = self.ecb3(out)
        out = self.esa(out)
        return out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    _, _, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale ** 2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True), nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0), nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class DCNv2Pack(ModulatedDeformConvPack):
    """Modulated deformable conv for deformable alignment.

    Different from the official DCNv2Pack, which generates offsets and masks
    from the preceding features, this DCNv2Pack takes another different
    features to generate offsets and masks.

    Ref:
        Delving Deep into Deformable Alignment in Video Super-Resolution.
    """

    def forward(self, x, feat):
        out = self.conv_offset(feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        offset_absmean = torch.mean(torch.abs(offset))
        if offset_absmean > 50:
            logger = get_root_logger()
            logger.warning(f'Offset abs mean is {offset_absmean}, larger than 50.')

        if LooseVersion(torchvision.__version__) >= LooseVersion('0.9.0'):
            return torchvision.ops.deform_conv2d(x, offset, self.weight, self.bias, self.stride, self.padding,
                                                 self.dilation, mask)
        else:
            return modulated_deform_conv(x, offset, mask, self.weight, self.bias, self.stride, self.padding,
                                         self.dilation, self.groups, self.deformable_groups)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        low = norm_cdf((a - mean) / std)
        up = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [low, up], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * low - 1, 2 * up - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution.

    From: https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/weight_init.py

    The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# From PyTorch
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple
