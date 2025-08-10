"""
RepViT classification backbone

This file implements the RepViT block stack used for efficient vision backbones.
Key ideas mirror those described in the RepViT family:
- Re-parameterizable depthwise conv branches ("RepVGGDW") for token mixing,
  fused into a single conv for inference to reduce latency.
- Channel mixing via pointwise MLP (1x1 convs) with residual connection.
- Optional squeeze-and-excitation (SE) for attention along channels.

Relevant references in docs/ (for architectural intuition and deployment):
- RepViT: Revisiting Mobile CNN From ViT Perspective (2023) — see sections on
  re-parameterization and efficient token/channel mixing.
- RepViT-SAM: Towards Real-Time Segmenting Anything (2023) — demonstrates using
  RepViT as the image encoder for SAM to lower latency while preserving accuracy.
- An Image Is Worth 16x16 Words (ViT, 2020) and Masked Autoencoders (MAE, 2021)
  provide background on tokenization and representation learning trends that inspired
  efficient token mixing in modern CNN/ViT hybrids.

Note: Only comments were added for documentation/citation; no functional code changes.
"""
import torch.nn as nn

def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

from timm.models.layers import SqueezeExcite

import torch

class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse Conv2d + BatchNorm2d into a single Conv2d for faster inference.
        This is a standard deployment trick discussed widely and also applied in
        RepViT/RepVGG-style re-parameterization. The resulting conv "bakes in"
        the BN affine and running-statistics.
        """
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups,
            device=c.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)
    
    @torch.no_grad()
    def fuse(self):
        """
        When the residual wrapper contains a depthwise Conv2d_BN (or Conv2d),
        fuse the branch and add identity via a kernel with center set to 1.0.
        This follows the re-parameterization principle used in RepVGG/RepViT,
        enabling the residual path to be represented as a single conv at deploy.
        See RepViT in docs for the deployment strategy.
        """
        if isinstance(self.m, Conv2d_BN):
            m = self.m.fuse()
            assert(m.groups == m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        elif isinstance(self.m, torch.nn.Conv2d):
            m = self.m
            assert(m.groups != m.in_channels)
            identity = torch.ones(m.weight.shape[0], m.weight.shape[1], 1, 1)
            identity = torch.nn.functional.pad(identity, [1,1,1,1])
            m.weight += identity.to(m.weight.device)
            return m
        else:
            return self


class RepVGGDW(torch.nn.Module):
    def __init__(self, ed) -> None:
        super().__init__()
        self.conv = Conv2d_BN(ed, ed, 3, 1, 1, groups=ed)
        self.conv1 = torch.nn.Conv2d(ed, ed, 1, 1, 0, groups=ed)
        self.dim = ed
        self.bn = torch.nn.BatchNorm2d(ed)
    
    def forward(self, x):
        return self.bn((self.conv(x) + self.conv1(x)) + x)
    
    @torch.no_grad()
    def fuse(self):
        """
        Fuse the 3x3 depthwise branch, the 1x1 depthwise branch, and identity
        into a single depthwise 3x3 conv, then fold the trailing BN. This is the
        core re-parameterization step that makes training-time multi-branch
        structure deploy as a single conv for low latency (see RepViT/RepVGG).
        """
        conv = self.conv.fuse()
        conv1 = self.conv1
        
        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias
        
        conv1_w = torch.nn.functional.pad(conv1_w, [1,1,1,1])

        identity = torch.nn.functional.pad(torch.ones(conv1_w.shape[0], conv1_w.shape[1], 1, 1, device=conv1_w.device), [1,1,1,1])

        final_conv_w = conv_w + conv1_w + identity
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        bn = self.bn
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = conv.weight * w[:, None, None, None]
        b = bn.bias + (conv.bias - bn.running_mean) * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        conv.weight.data.copy_(w)
        conv.bias.data.copy_(b)
        return conv


class RepViTBlock(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(RepViTBlock, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup
        assert(hidden_dim == 2 * inp)

        if stride == 2:
            # Downsampling stage: use depthwise token mixer (spatial conv) possibly
            # with SE, then pointwise channel mixer (MLP via 1x1 convs). This mirrors
            # the token/channel split in RepViT blocks as described in the paper.
            self.token_mixer = nn.Sequential(
                Conv2d_BN(inp, inp, kernel_size, stride, (kernel_size - 1) // 2, groups=inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
                Conv2d_BN(inp, oup, ks=1, stride=1, pad=0)
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(oup, 2 * oup, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(2 * oup, oup, 1, 1, 0, bn_weight_init=0),
                ))
        else:
            assert(self.identity)
            # Identity stage: RepVGGDW provides a re-parameterizable token mixer
            # with an implicit identity path, followed by channel MLP. During
            # deployment, the RepVGGDW block can be fused to a single conv.
            self.token_mixer = nn.Sequential(
                RepVGGDW(inp),
                SqueezeExcite(inp, 0.25) if use_se else nn.Identity(),
            )
            self.channel_mixer = Residual(nn.Sequential(
                    # pw
                    Conv2d_BN(inp, hidden_dim, 1, 1, 0),
                    nn.GELU() if use_hs else nn.GELU(),
                    # pw-linear
                    Conv2d_BN(hidden_dim, oup, 1, 1, 0, bn_weight_init=0),
                ))

    def forward(self, x):
        return self.channel_mixer(self.token_mixer(x))

from timm.models.vision_transformer import trunc_normal_
class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        """
        Fuse BatchNorm1d into Linear by folding affine params and statistics.
        This mirrors the Conv-BN folding used for conv layers and simplifies
        deployment graphs.
        """
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0), device=l.weight.device)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

class Classfier(nn.Module):
    def __init__(self, dim, num_classes, distillation=True):
        super().__init__()
        self.classifier = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.classifier_dist = BN_Linear(dim, num_classes) if num_classes > 0 else torch.nn.Identity()

    def forward(self, x):
        if self.distillation:
            x = self.classifier(x), self.classifier_dist(x)
            if not self.training:
                x = (x[0] + x[1]) / 2
        else:
            x = self.classifier(x)
        return x

    @torch.no_grad()
    def fuse(self):
        """
        If distillation was used, average the fused heads for evaluation
        consistency, following standard practice in DeiT-style distillation.
        """
        classifier = self.classifier.fuse()
        if self.distillation:
            classifier_dist = self.classifier_dist.fuse()
            classifier.weight += classifier_dist.weight
            classifier.bias += classifier_dist.bias
            classifier.weight /= 2
            classifier.bias /= 2
            return classifier
        else:
            return classifier

class RepViT(nn.Module):
    def __init__(self, cfgs, num_classes=1000, distillation=False):
        super(RepViT, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs

        # building first layer
        # Patch embedding: two strided convs to quickly reduce spatial resolution,
        # analogous to ViT patchify but using convs for efficiency (cf. RepViT).
        input_channel = self.cfgs[0][2]
        patch_embed = torch.nn.Sequential(Conv2d_BN(3, input_channel // 2, 3, 2, 1), torch.nn.GELU(),
                           Conv2d_BN(input_channel // 2, input_channel, 3, 2, 1))
        layers = [patch_embed]
        # building inverted residual blocks
        # Each cfg entry: [kernel_size, expansion, out_channels, use_se, use_hs, stride]
        block = RepViTBlock
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.ModuleList(layers)
        self.classifier = Classfier(output_channel, num_classes, distillation)
        
    def forward(self, x):
        # x = self.features(x)
        for f in self.features:
            x = f(x)
        x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        x = self.classifier(x)
        return x

from timm.models import register_model


@register_model
def repvit_m0_6(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        [3,   2,  40, 1, 0, 1],
        [3,   2,  40, 0, 0, 1],
        [3,   2,  80, 0, 0, 2],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 1, 2],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 1, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 160, 0, 1, 1],
        [3,   2, 320, 0, 1, 2],
        [3,   2, 320, 1, 1, 1],
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m0_9(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  48, 1, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  48, 0, 0, 1],
        [3,   2,  96, 0, 0, 2],
        [3,   2,  96, 1, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  96, 0, 0, 1],
        [3,   2,  192, 0, 1, 2],
        [3,   2,  192, 1, 1, 1],
        [3,   2,  192, 0, 1, 1],
        [3,   2,  192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 1, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 192, 0, 1, 1],
        [3,   2, 384, 0, 1, 2],
        [3,   2, 384, 1, 1, 1],
        [3,   2, 384, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)

@register_model
def repvit_m1_0(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  56, 1, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  56, 0, 0, 1],
        [3,   2,  112, 0, 0, 2],
        [3,   2,  112, 1, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  112, 0, 0, 1],
        [3,   2,  224, 0, 1, 2],
        [3,   2,  224, 1, 1, 1],
        [3,   2,  224, 0, 1, 1],
        [3,   2,  224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 1, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 224, 0, 1, 1],
        [3,   2, 448, 0, 1, 2],
        [3,   2, 448, 1, 1, 1],
        [3,   2, 448, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m1_1(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)


@register_model
def repvit_m1_5(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 1, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  64, 0, 0, 1],
        [3,   2,  128, 0, 0, 2],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 1, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  128, 0, 0, 1],
        [3,   2,  256, 0, 1, 2],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2,  256, 0, 1, 1],
        [3,   2,  256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 1, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 256, 0, 1, 1],
        [3,   2, 512, 0, 1, 2],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1],
        [3,   2, 512, 1, 1, 1],
        [3,   2, 512, 0, 1, 1]
    ]
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)



@register_model
def repvit_m2_3(pretrained=False, num_classes = 1000, distillation=False):
    """
    Constructs a MobileNetV3-Large model
    """
    cfgs = [
        # k, t, c, SE, HS, s 
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 1, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  80, 0, 0, 1],
        [3,   2,  160, 0, 0, 2],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 1, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  160, 0, 0, 1],
        [3,   2,  320, 0, 1, 2],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2,  320, 0, 1, 1],
        [3,   2,  320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 1, 1, 1],
        [3,   2, 320, 0, 1, 1],
        # [3,   2, 320, 1, 1, 1],
        # [3,   2, 320, 0, 1, 1],
        [3,   2, 320, 0, 1, 1],
        [3,   2, 640, 0, 1, 2],
        [3,   2, 640, 1, 1, 1],
        [3,   2, 640, 0, 1, 1],
        # [3,   2, 640, 1, 1, 1],
        # [3,   2, 640, 0, 1, 1]
    ]    
    return RepViT(cfgs, num_classes=num_classes, distillation=distillation)