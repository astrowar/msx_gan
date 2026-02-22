import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Quantização Customizada de 2/4 bits (16/4 níveis) - STE fake-quant
# ============================================================================

class FourBitQuantizer(nn.Module):
    def __init__(self, symmetric=True, n_bits=4):
        super(FourBitQuantizer, self).__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.symmetric = symmetric

        self.scale = nn.Parameter(torch.tensor(1.0))
        if not symmetric:
            self.zero_point = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('zero_point', torch.tensor(0.0))

    def forward(self, x, quant_temp=1.0):
        if not self.training:
            return self.quantize(x)
        return self.fake_quantize(x, quant_temp)

    def quantize(self, x):
        scale = self.scale.abs() + 1e-12
        if self.symmetric:
            x_clamped = torch.clamp(x, -scale, scale)
            x_normalized = (x_clamped / scale + 1) / 2
        else:
            x_clamped = torch.clamp(x, self.zero_point, self.zero_point + scale)
            x_normalized = (x_clamped - self.zero_point) / scale

        x_quant = torch.round(x_normalized * (self.n_levels - 1))

        x_dequant = x_quant / (self.n_levels - 1)
        if self.symmetric:
            x_dequant = (x_dequant * 2 - 1) * scale
        else:
            x_dequant = x_dequant * scale + self.zero_point

        return x_dequant

    def fake_quantize(self, x, quant_temp=1.0):
        scale = self.scale.abs() + 1e-12
        if self.symmetric:
            x_clamped = torch.clamp(x, -scale, scale)
            x_normalized = (x_clamped / scale + 1) / 2
        else:
            x_clamped = torch.clamp(x, self.zero_point, self.zero_point + scale)
            x_normalized = (x_clamped - self.zero_point) / scale

        x_quant = torch.round(x_normalized * (self.n_levels - 1))

        x_dequant = x_quant / (self.n_levels - 1)
        if self.symmetric:
            x_dequant = (x_dequant * 2 - 1) * scale
        else:
            x_dequant = x_dequant * scale + self.zero_point

        x_mixed = quant_temp * x_dequant + (1.0 - quant_temp) * x
        return x + (x_mixed - x).detach()


# class QuantizedConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, act_symmetric=False, n_bits=4):
#         super().__init__()
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
#         self.weight_quantizer = FourBitQuantizer(symmetric=True, n_bits=n_bits)
#         self.activation_quantizer = FourBitQuantizer(symmetric=act_symmetric, n_bits=n_bits)

#     def forward(self, x, quant_temp=1.0):
#         xq = self.activation_quantizer(x, quant_temp)
#         wq = self.weight_quantizer(self.conv.weight, quant_temp)
#         return F.conv2d(xq, wq, self.conv.bias, self.conv.stride, self.conv.padding)


class QuantizedConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, act_symmetric=False, n_bits=4):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.weight_quantizer = FourBitQuantizer(symmetric=True, n_bits=n_bits)
        self.activation_quantizer = FourBitQuantizer(symmetric=act_symmetric, n_bits=n_bits)

    def forward(self, x, quant_temp=1.0):
        xq = self.activation_quantizer(x, quant_temp)
        wq = self.weight_quantizer(self.conv.weight, quant_temp)
        return F.conv_transpose2d(xq, wq, self.conv.bias, self.conv.stride, self.conv.padding)


# ============================================================================
# Fold BN into ConvTranspose2d (para export/checkpoint)
# ============================================================================

@torch.no_grad()
def fold_bn_into_convT(convT: nn.ConvTranspose2d, bn: nn.BatchNorm2d):
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    invstd = torch.rsqrt(var + eps)
    s = gamma * invstd

    # ConvTranspose2d weight: [in_channels, out_channels, kH, kW]
    W = convT.weight
    W_fold = W * s.view(1, -1, 1, 1)

    if convT.bias is None:
        b = torch.zeros_like(mean)
    else:
        b = convT.bias

    b_fold = (b - mean) * s + beta

    convT.weight.copy_(W_fold)
    if convT.bias is None:
        convT.bias = nn.Parameter(b_fold.clone())
    else:
        convT.bias.copy_(b_fold)


@torch.no_grad()
def fold_generator_bn_32(netG: nn.Module):
    # Fold BN onde existe BN imediatamente após a convT (deconv1/2/3)
    fold_bn_into_convT(netG.deconv1.conv, netG.bn1)
    fold_bn_into_convT(netG.deconv2.conv, netG.bn2)
    fold_bn_into_convT(netG.deconv3.conv, netG.bn3)

    netG.bn1 = nn.Identity()
    netG.bn2 = nn.Identity()
    netG.bn3 = nn.Identity()


@torch.no_grad()
def fold_generator_bn_32_quad(netG: nn.Module):
    fold_bn_into_convT(netG.deconv1.conv, netG.bn1)
    fold_bn_into_convT(netG.deconv2.conv, netG.bn2)

    fold_bn_into_convT(netG.deconv3_tl.conv, netG.bn3_tl)
    fold_bn_into_convT(netG.deconv3_tr.conv, netG.bn3_tr)
    fold_bn_into_convT(netG.deconv3_bl.conv, netG.bn3_bl)
    fold_bn_into_convT(netG.deconv3_br.conv, netG.bn3_br)

    netG.bn1 = nn.Identity()
    netG.bn2 = nn.Identity()
    netG.bn3_tl = nn.Identity()
    netG.bn3_tr = nn.Identity()
    netG.bn3_bl = nn.Identity()
    netG.bn3_br = nn.Identity()


# ============================================================================
# Generator (32x32) - 4 layers, todos kernels 4x4
# Geometria: z->2x2 -> 4 -> 8 -> 16 -> 32
# Condição: nz == nw*4
# ============================================================================

class QuantizedGenerator32(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nw, n_bits=4):
        super().__init__()
        self.ngpu = ngpu
        self.nw = nw
        assert nz == nw * 4, f"Para reshape 2x2 precisa nz == nw*4. Recebido nz={nz}, nw={nw}"

        c1 = ngf * 8
        c2 = ngf * 4
        c3 = max(4, ngf)

        # deconv1 recebe z (negativos) -> ativação simétrica
        self.deconv1 = QuantizedConvTranspose2d(nw, c1, 4, 2, 1, bias=False, act_symmetric=True, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(True)

        self.deconv2 = QuantizedConvTranspose2d(c1, c2, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(True)

        self.deconv3 = QuantizedConvTranspose2d(c2, c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(True)

        self.deconv4 = QuantizedConvTranspose2d(c3, nc, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.tanh = nn.Tanh()

    def forward(self, z, quant_temp=1.0):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))
        x = z.view(z.size(0), self.nw, 2, 2)

        x = self.deconv1(x, quant_temp)  # 4x4
        x = self.bn1(x); x = self.relu1(x)

        x = self.deconv2(x, quant_temp)  # 8x8
        x = self.bn2(x); x = self.relu2(x)

        x = self.deconv3(x, quant_temp)  # 16x16
        x = self.bn3(x); x = self.relu3(x)

        x = self.deconv4(x, quant_temp)  # 32x32
        return self.tanh(x)


# ============================================================================
# Generator (24x24) - 4 layers k=4, s=2 (padding diferente no layer 2)
# Geometria: z->2x2 -> 4 -> 6 -> 12 -> 24
# Condição: nz == nw*4 (ex: nz=64, nw=16)
# Kernel SEMPRE 4x4 - apenas padding muda no layer 2
# 3 layers idênticos (k=4,s=2,p=1) + 1 layer especial (k=4,s=2,p=2)
# ============================================================================

class QuantizedGenerator24_original(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nw, n_bits=4):
        super().__init__()
        self.ngpu = ngpu
        self.nw = nw
        assert nz == nw * 4, f"Para reshape 2x2 precisa nz == nw*4. Recebido nz={nz}, nw={nw}"

        c1 = ngf * 8
        c2 = ngf * 4
        c3 = max(4, ngf)

        # Layer 1: 2->4 (k=4, s=2, p=1) - PADRÃO
        self.deconv1 = QuantizedConvTranspose2d(nw, c1, 4, 2, 1, bias=False, act_symmetric=True, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(True)

        # Layer 2: 4->6 (k=4, s=2, p=2) - PADDING DIFERENTE
        self.deconv2 = QuantizedConvTranspose2d(c1, c2, 4, 2, 2, bias=False, act_symmetric=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(True)

        # Layer 3: 6->12 (k=4, s=2, p=1) - PADRÃO
        self.deconv3 = QuantizedConvTranspose2d(c2, c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(True)

        # Layer 4: 12->24 (k=4, s=2, p=1) - PADRÃO
        self.deconv4 = QuantizedConvTranspose2d(c3, nc, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.tanh = nn.Tanh()

    def forward(self, z, quant_temp=1.0):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))
        x = z.view(z.size(0), self.nw, 2, 2)  # reshape para 2x2

        x = self.deconv1(x, quant_temp)  # 2x2 -> 4x4
        x = self.bn1(x); x = self.relu1(x)

        x = self.deconv2(x, quant_temp)  # 4x4 -> 6x6 (padding especial)
        x = self.bn2(x); x = self.relu2(x)

        x = self.deconv3(x, quant_temp)  # 6x6 -> 12x12
        x = self.bn3(x); x = self.relu3(x)

        x = self.deconv4(x, quant_temp)  # 12x12 -> 24x24
        return self.tanh(x)




class GeneratorMono24_Simple(nn.Module):
    """
    Gerador simples:
      z (B, nz) com nz = nw*4
      reshape -> (B, nw, 2, 2)
      upsample nearest: 2->6 (x3), 6->12 (x2), 12->24 (x2)
      conv3x3 + ReLU em tudo, exceto última layer (useRELU=False)
      saída mono (1, 24, 24)
    """
    def __init__(self, nz, nw=16, ch_plan=(24, 22, 6, 5, 5)):
        super().__init__()
        self.nw = nw
        assert nz == nw * 4, f"Precisa nz == nw*4. Recebido nz={nz}, nw={nw}"

        c2, c6, c12, c24, cH = ch_plan

        self.l0 = Layer(nw,  c2,  3, 1, 1, bias=False, useRELU=True)   # 2x2
        self.l1 = Layer(c2,  c6,  3, 1, 1, bias=False, useRELU=True)   # 6x6
        self.l2 = Layer(c6,  c12, 3, 1, 1, bias=False, useRELU=True)   # 12x12
        self.l3 = Layer(c12, c24, 3, 1, 1, bias=False, useRELU=True)   # 24x24
        self.l4 = Layer(c24, cH,  3, 1, 1, bias=False, useRELU=True)   # refino 24x24

        # saída mono, SEM ReLU (pra não cortar negativos antes do tanh)
        self.out = Layer(cH, 1,   3, 1, 1, bias=True,  useRELU=False)

        self.tanh = nn.Tanh()

    @staticmethod
    def up_to(x, h, w):
        return F.interpolate(x, size=(h, w), mode="nearest")

    def forward(self, z):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))

        x = z.view(z.size(0), self.nw, 2, 2)

        x = self.l0(x)                 # 2x2

        x = self.up_to(x, 6, 6)        # x3
        x = self.l1(x)                 # 6x6

        x = self.up_to(x, 12, 12)      # x2
        x = self.l2(x)                 # 12x12

        x = self.up_to(x, 24, 24)      # x2
        x = self.l3(x)                 # 24x24

        x = self.l4(x)                 # refino
        x = self.out(x)                # logits mono
        return self.tanh(x)


# ============================================================================
# Discriminator (24x24) - downsample layers
# Geometria: 24 -> 12 -> 6 -> 3 -> 1
# ============================================================================
class Layer(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, *, bias=False, useRELU=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.useRELU = useRELU
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        if self.useRELU:
            x = self.relu(x)
        return x


class DLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=2, p=1, *, bias=True, useLRELU=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.useLRELU = useLRELU
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.useLRELU:
            x = self.act(x)
        return x


class Discriminator24(nn.Module):
    """
    Entrada: (B,1,H,W) mono
    Saída: (B,) logit/score (sem sigmoid) -> use Hinge ou WGAN-GP
    Funciona com 24x24 e outros tamanhos.
    """
    def __init__(self, ndf=32):
        super().__init__()
        self.l1 = DLayer(1,    ndf,   3, 2, 1)      # /2
        self.l2 = DLayer(ndf,  ndf*2, 3, 2, 1)      # /2
        self.l3 = DLayer(ndf*2,ndf*4, 3, 2, 1)      # /2
        self.l4 = DLayer(ndf*4,ndf*4, 3, 1, 1)      # refino (stride=1)

        # head 1x1: vira 1 canal
        self.head = nn.Conv2d(ndf*4, 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x ):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.head(x)                 # (B,1,h,w)
        x = x.mean(dim=(2,3))            # global average -> (B,1)
        return x.view(x.size(0))         # (B,)

# ============================================================================
class QuantizedGenerator32_QuadL3(nn.Module):
    def __init__(self, ngpu, nz, ngf, nc, nw, n_bits=4):
        super().__init__()
        self.ngpu = ngpu
        self.nw = nw
        assert nz == nw * 4, "Para 32x32 com reshape 2x2: nz == nw*4"

        self.c1 = ngf * 8
        self.c2 = ngf * 4
        self.c3 = max(4, ngf)

        # Trunk
        self.deconv1 = QuantizedConvTranspose2d(nw, self.c1, 4, 2, 1, bias=False, act_symmetric=True, n_bits=n_bits)
        self.bn1 = nn.BatchNorm2d(self.c1)
        self.deconv2 = QuantizedConvTranspose2d(self.c1, self.c2, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.bn2 = nn.BatchNorm2d(self.c2)

        self.relu = nn.ReLU(True)

        # Heads
        self.deconv3_tl = QuantizedConvTranspose2d(self.c2, self.c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.deconv3_tr = QuantizedConvTranspose2d(self.c2, self.c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.deconv3_bl = QuantizedConvTranspose2d(self.c2, self.c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.deconv3_br = QuantizedConvTranspose2d(self.c2, self.c3, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)

        self.bn3_tl = nn.BatchNorm2d(self.c3)
        self.bn3_tr = nn.BatchNorm2d(self.c3)
        self.bn3_bl = nn.BatchNorm2d(self.c3)
        self.bn3_br = nn.BatchNorm2d(self.c3)

        # Last global
        self.deconv4 = QuantizedConvTranspose2d(self.c3, nc, 4, 2, 1, bias=False, act_symmetric=False, n_bits=n_bits)
        self.tanh = nn.Tanh()

    @staticmethod
    def _blend_copy(dst, src, y0, x0, blend_mask=None):
        h, w = src.shape[-2], src.shape[-1]
        region = dst[:, :, y0:y0 + h, x0:x0 + w]
        if blend_mask is None:
            dst[:, :, y0:y0 + h, x0:x0 + w] = src
        else:
            dst[:, :, y0:y0 + h, x0:x0 + w] = region * (1.0 - blend_mask) + src * blend_mask

    def forward(self, z, quant_temp=1.0):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))
        x = z.view(z.size(0), self.nw, 2, 2)

        x = self.deconv1(x, quant_temp)
        x = self.bn1(x); x = self.relu(x)

        l2 = self.deconv2(x, quant_temp)
        l2 = self.bn2(l2); l2 = self.relu(l2)  # [B, c2, 8, 8]

        def mk_tile(deconv3, bn3):
            t = deconv3(l2, quant_temp)   # [B,c3,16,16]
            t = bn3(t); t = self.relu(t)
            t_pad = F.pad(t, (1, 1, 1, 1), mode="replicate")  # 18x18
            return t_pad

        tl_pad = mk_tile(self.deconv3_tl, self.bn3_tl)
        tr_pad = mk_tile(self.deconv3_tr, self.bn3_tr)
        bl_pad = mk_tile(self.deconv3_bl, self.bn3_bl)
        br_pad = mk_tile(self.deconv3_br, self.bn3_br)

        L3 = torch.zeros((z.size(0), self.c3, 16, 16), device=z.device, dtype=tl_pad.dtype)

        def mask_9x9(blend_left=False, blend_top=False):
            m = torch.ones((1, 1, 9, 9), device=z.device, dtype=tl_pad.dtype)
            if blend_left:
                m[:, :, :, 0:1] = 0.5
            if blend_top:
                m[:, :, 0:1, :] = 0.5
            return m

        tl = tl_pad[:, :, 0:9, 0:9]
        tr = tr_pad[:, :, 0:9, 9:18]
        bl = bl_pad[:, :, 9:18, 0:9]
        br = br_pad[:, :, 9:18, 9:18]

        self._blend_copy(L3, tl, 0, 0, blend_mask=None)
        self._blend_copy(L3, tr, 0, 7, blend_mask=mask_9x9(blend_left=True, blend_top=False))
        self._blend_copy(L3, bl, 7, 0, blend_mask=mask_9x9(blend_left=False, blend_top=True))
        self._blend_copy(L3, br, 7, 7, blend_mask=mask_9x9(blend_left=True, blend_top=True))

        out = self.deconv4(L3, quant_temp)
        return self.tanh(out)


# ============================================================================
# Discriminator (32x32) - BCEWithLogits (sem Sigmoid no modelo)
# 32 -> 16 -> 8 -> 4 -> 1 usando kernels 4x4
# ============================================================================

class Discriminator32(nn.Module):
    def __init__(self, ngpu, nc=1, ndf=12, use_sigmoid=False):
        """
        Discriminator para imagens 32x32.

        Args:
            ngpu: número de GPUs
            nc: número de canais de entrada
            ndf: número de features do discriminator
            use_sigmoid: se True, aplica sigmoid na saída (para BCE).
                        se False, saída linear (para WGAN)
        """
        super().__init__()
        self.ngpu = ngpu
        self.use_sigmoid = use_sigmoid

        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),          # 32 -> 16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),     # 16 -> 8
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False), # 8 -> 4
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),       # 4 -> 1 (logit)
        )

    def forward(self, x):
        out = self.main(x)
        out = out.view(-1, 1).squeeze(1)

        if self.use_sigmoid:
            out = torch.sigmoid(out)

        return out


# ============================================================================
# Inicialização de pesos
# ============================================================================

def weights_init(m):

        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, "weight"):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)
