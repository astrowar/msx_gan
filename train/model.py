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


#quantizer for 8bits
class EightBitQuantize(nn.Module):
    def __init__(self, symmetric=True, n_bits=8):
        super(EightBitQuantize, self).__init__()
        self.symmetric = symmetric

        # support variable number of bits (keeps backward compat with 8-bit)
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits

        self.scale = nn.Parameter(torch.tensor(1.0))
        if not symmetric:
            self.zero_point = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('zero_point', torch.tensor(0.0))

    def forward(self, x, quant_temp=0.0):
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

    def fake_quantize(self, x, quant_temp=0.0):
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



class QuantizedConv(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, *, bias=True, act_symmetric=False, n_bits=4):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.act_symmetric = act_symmetric
        self.n_bits = n_bits

        self.quantizer = EightBitQuantize(symmetric=act_symmetric, n_bits=n_bits)

    def forward(self, x, quant_temp=1.0):
        x = self.conv(x)
        x = self.quantizer(x, quant_temp)
        return x


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
def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d):
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps

    invstd = torch.rsqrt(var + eps)
    s = gamma * invstd

    # Conv2d weight: [out_channels, in_channels, kH, kW]
    W = conv.weight
    W_fold = W * s.view(-1, 1, 1, 1)

    if conv.bias is None:
        b = torch.zeros_like(mean)
    else:
        b = conv.bias

    b_fold = (b - mean) * s + beta

    conv.weight.copy_(W_fold)
    if conv.bias is None:
        conv.bias = nn.Parameter(b_fold.clone())
    else:
        conv.bias.copy_(b_fold)


@torch.no_grad()
def adjust_layer(layer: nn.Module, scale: float):
    if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d)):
        layer.weight.mul_(scale)
        if layer.bias is not None:
            layer.bias.mul_(scale)
    elif isinstance(layer, QuantizedConv):
        layer.conv.weight.mul_(scale)
        if layer.conv.bias is not None:
            layer.conv.bias.mul_(scale)
    else:
        raise TypeError(f"Layer must be Conv2d/ConvTranspose2d, got {type(layer)}")


@torch.no_grad()
def get_layer_stat(flat: torch.Tensor, mode="99"):
    x = flat.detach().float().abs()
    if mode == "max":
        return float(x.max().item())
    elif mode == "p99":
        return float(x.quantile(0.99).item())
    elif mode == "p75":
        return float(x.quantile(0.75).item())
    else:
        raise ValueError("mode must be max/p99/p75")

@torch.no_grad()
def compute_scale(cur: float, target: float, alpha: float,  min_scale: float, max_scale: float):
    # queremos: cur * s ~= target  => s = target/cur
    if cur < 1e-12:
        raw = 1.0
    else:
        raw = target / cur

    # aplica só uma fração pra não explodir (no log espaço é melhor)
    # s_applied = raw^alpha  (suave e estável)
    s = float(raw ** alpha)

    # clamp
    if s < min_scale: s = min_scale
    if s > max_scale: s = max_scale
    return s, raw



@torch.no_grad()
def adjust_layer_amplitude_sequential(
    netG: nn.Module,
    z: torch.Tensor,
    mode="p99",             # "max" ou "p99" ou "p999"
    target_relu=64.0,        # alvo para camadas ReLU
    target_out=64.0,         # alvo para camada out (pré-tanh)
    alpha=0.2,             # fração de correção (0.1~0.3 bom)
    min_scale=0.8,          # limita mudança por passo
    max_scale=1.2,
    verbose=True,
):
    """
    Ajusta as camadas em sequência, recalculando ranges a cada etapa.
    Requer netG.collectRanges(z, quant_temp=0.0) retornando:
      l0_flat, l1_flat, l2_flat, l3_flat, l4_flat, out_flat
    """

    def collect():
        l0_flat, l1_flat, l2_flat, l3_flat, l4_flat, out_flat = netG.collectRanges(z, quant_temp=0.0)
        return {
            "l0": l0_flat,
            "l1": l1_flat,
            "l2": l2_flat,
            "l3": l3_flat,
            "l4": l4_flat,
            "out": out_flat,
        }

    # mapeia para os convs reais (ajuste conforme seu modelo)
    conv_map = {
        "l0": netG.l0.conv,
        "l1": netG.l1.conv,
        "l2": netG.l2.conv,
        "l3": netG.l3.conv,
        "l4": netG.l4.conv,
        "out": netG.out.conv,
    }

    # 0) stats iniciais
    stats = collect()
    if verbose:
        print("BEFORE:")
        for k in ["l0","l1","l2","l3","l4","out"]:
            cur = get_layer_stat(stats[k], mode=mode)
            print(f"  {k}: {mode}={cur:.6f}")

    # 1) ajusta sequencialmente (recalcula após cada ajuste)
    for k in ["l0","l1","l2","l3","l4","out"]:
        stats = collect()
        cur = get_layer_stat(stats[k], mode=mode)

        target = target_out if k == "out" else target_relu

        if cur > 30 and cur < 90:
            target =  1.0* cur
        
        s_applied, s_raw = compute_scale(cur, target, alpha, min_scale, max_scale)

        if verbose:
            print(f"[{k}] cur={cur:.6f} target={target:.6f} raw={s_raw:.4f} applied={s_applied:.4f}")

        adjust_layer(conv_map[k], s_applied)

    # 2) stats finais
    stats = collect()
    if verbose:
        print("AFTER:")
        for k in ["l0","l1","l2","l3","l4","out"]:
            cur = get_layer_stat(stats[k], mode=mode)
            print(f"  {k}: {mode}={cur:.6f}")

    return netG




# ============================================================================
# Generator (24x24) - 4 layers k=4, s=2 (padding diferente no layer 2)
# Geometria: z->2x2 -> 4 -> 6 -> 12 -> 24
# Condição: nz == nw*4 (ex: nz=64, nw=16)
# Kernel SEMPRE 4x4 - apenas padding muda no layer 2
# 3 layers idênticos (k=4,s=2,p=1) + 1 layer especial (k=4,s=2,p=2)
# ============================================================================
class GeneratorMono24_Simple(nn.Module):
    """
    Gerador simples:
      z (B, nz) com nz = nw*4
      reshape -> (B, nw, 2, 2)
      upsample nearest: 2->6 (x3), 6->12 (x2), 12->24 (x2)
      conv3x3 + ReLU em tudo, exceto última layer (useRELU=False)
      saída mono (1, 24, 24)
    """
    def __init__(self, nz, nw=16, ch_plan=(22, 22, 6, 6,4)):
        super().__init__()
        self.nw = nw
        assert nz == nw * 4, f"Precisa nz == nw*4. Recebido nz={nz}, nw={nw}"

        c2, c6, c12, c24, cH = ch_plan

        self.l0 = Layer(nw,  c2,  3, 1, 1, bias=False, useRELU=True )   # 2x2
        self.l1 = Layer(c2,  c6,  3, 1, 1, bias=False, useRELU=True )   # 6x6
        self.l2 = Layer(c6,  c12, 3, 1, 1, bias=False, useRELU=True )   # 12x12
        self.l3 = Layer(c12, c24, 3, 1, 1, bias=False, useRELU=True )   # 24x24
        self.l4 = Layer(c24, cH,  3, 1, 1, bias=False, useRELU=True )   # refino 24x24

        # saída mono, SEM ReLU (pra não cortar negativos antes do tanh)
        # don't apply BatchNorm on final output layer
        self.out = Layer(cH, 1,   3, 1, 1, bias=True,  useRELU=False )

        self.tanh = nn.Tanh()
        
        self.sigmoid = nn.Sigmoid()
        
        self.noise = lambda: 0.5*torch.randn(1, device=next(self.parameters()).device)

       

    @staticmethod
    def up_to(x, h, w):
        return F.interpolate(x, size=(h, w), mode="nearest")

    def forward(self, z, quant_temp):
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))

        x = z.view(z.size(0), self.nw, 2, 2)

        x =x * 16
        x = self.l0(x, quant_temp)                 # 2x2  
        x = 64*self.tanh(x/64) +  self.noise()
            

        x = self.up_to(x, 6, 6)        # x3
        x = self.l1(x, quant_temp)    
        x= 64*self.tanh(x/64) +   self.noise()               # 6x6
  

        x = self.up_to(x, 12, 12)      # x2
        x = self.l2(x, quant_temp)                 # 12x12    
        x = 64*self.tanh(x/64) +    self.noise()              # 12x12

        x = self.up_to(x, 24, 24)      # x2
        x = self.l3(x, quant_temp)                 # 24x24  
        x =  64*self.tanh(x/64) +    self.noise()           # 24x24   


        x = self.l4(x, quant_temp)                 # refino    
        x = 64*self.tanh(x/64) +  self.noise() 
        
        
        x = self.out(x)                # logits mono


        return self.tanh(x/16)                # saída em [-1,1]
        #return self.sigmoid(x)                # saída em [0,1]


    def collectRanges(self, z, quant_temp=0.0):
        # Passa o input pela rede, coletando os ranges de cada layer e escrevendo na tela para debug/monitoramento
        # ie cada layer output , Mean, min(P99) and max(P99) dos outputs, pra ver se tá saturando ou se tem espaço de quantização
        #recebe um batch em x [:B, nz] ou [B, nw, 2, 2]
        if z.dim() == 4:
            z = z.view(z.size(0), z.size(1))

        x = z.view(z.size(0), self.nw, 2, 2)

        x =x  * 16

        x = self.l0(x, quant_temp)                 # 2x2
        l0_flat = x.detach().view(-1)
        x = self.up_to(x, 6, 6)        # x3
        x = self.l1(x, quant_temp)                 # 6x6
        l1_flat = x.detach().view(-1)

        x = self.up_to(x, 12, 12)      # x2
        x = self.l2(x, quant_temp)                 # 12x12
        l2_flat = x.detach().view(-1)

        x = self.up_to(x, 24, 24)      # x2
        x = self.l3(x, quant_temp)                 # 24x24
        l3_flat = x.detach().view(-1)

        x = self.l4(x, quant_temp)                 # refino
        l4_flat = x.detach().view(-1)
        x = self.out(x)                # logits mono
        out_flat = x.detach().view(-1)

        return l0_flat, l1_flat, l2_flat, l3_flat, l4_flat, out_flat


class Layer(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, *, bias=False, useRELU=True ):
        super().__init__()
        self.conv =nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.useRELU = useRELU
        self.relu = nn.ReLU(True) 

    def forward(self, x , quant_temp = 0.0 ): 
        x = self.conv(x)
        if self.useRELU:
            x = self.relu(x)
        return x
class DLayer(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, *, bias=True, use_act=True, negative_slope=0.2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.use_act = use_act
        self.act = nn.LeakyReLU(negative_slope, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = self.act(x)
        return x


class CriticMono24_Simple(nn.Module):
    """
    Crítico/Discriminador simples para 24x24 mono:
      entrada: (B, 1, 24, 24) em [-1,1] (saída do tanh do gerador)
      blocos conv3x3 + LeakyReLU
      downsample explícito: 24->12->6->2
      saída: score escalar (B,)
    """
    def __init__(self, ch_plan=(4, 6, 6, 22, 22), negative_slope=0.2):
        super().__init__()
        c24a, c12, c6, c2, cH = ch_plan

        # 24x24
        self.l0 = DLayer(1,    c24a, 3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)
        self.l1 = DLayer(c24a, c24a, 3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)

        # 12x12
        self.l2 = DLayer(c24a, c12,  3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)
        self.l3 = DLayer(c12,  c12,  3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)

        # 6x6
        self.l4 = DLayer(c12,  c6,   3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)
        self.l5 = DLayer(c6,   c6,   3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)

        # 2x2 (fazemos resize direto 6->2, espelhando a ideia do G de resize explícito)
        self.l6 = DLayer(c6,   c2,   3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)
        self.l7 = DLayer(c2,   cH,   3, 1, 1, bias=True, use_act=True, negative_slope=negative_slope)

        # cabeça final: cH x 2 x 2 -> 1 score
        self.head = nn.Conv2d(cH, 1, kernel_size=2, stride=1, padding=0, bias=True)

    @staticmethod
    def down_to(x, h, w):
        # nearest também funciona, mas avg costuma ser mais estável no crítico
        return F.interpolate(x, size=(h, w), mode="area")

    def forward(self, x):
        # garante shape (B,1,24,24)
        if x.dim() == 3:
            x = x.unsqueeze(1)


        # 24x24
        x = self.l0(x)
        x = self.l1(x)

        # 24 -> 12
        x = self.down_to(x, 12, 12)
        x = self.l2(x)
        x = self.l3(x)

        # 12 -> 6
        x = self.down_to(x, 6, 6)
        x = self.l4(x)
        x = self.l5(x)

        # 6 -> 2
        x = self.down_to(x, 2, 2)
        x = self.l6(x)
        x = self.l7(x)

        # (B, cH, 2, 2) -> (B,1,1,1)
        x = self.head(x)

        return x.view(x.size(0))  # score escalar por amostra (WGAN critic)

def weights_init(m):

        classname = m.__class__.__name__
        if classname.find('Conv') != -1 and hasattr(m, "weight"):
            torch.nn.init.normal_(m.weight, 0.0, 0.12)
        elif classname.find('ConvTranspose') != -1 and hasattr(m, "weight"):
            torch.nn.init.normal_(m.weight, 0.0, 0.12)


        elif classname.find('BatchNorm') != -1:
            torch.nn.init.normal_(m.weight, 1.0, 0.02)
            torch.nn.init.zeros_(m.bias)
