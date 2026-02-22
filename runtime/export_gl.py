import math
import numpy as np
import torch
import torch.nn as nn
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# ------------------------------------------------------------
# Ajuste aqui conforme seu gerador
# ------------------------------------------------------------
from model import GeneratorMono24_Simple  # <-- troque

MODEL_PATH = "/home/astro/ml/gan/c32/train/output_wgan_4bit_24/netG_wgan_epoch_360.pth"          # state_dict do gerador
OUT_H      = "gen_weights.h"  # header gerado



NW = 16
NZ = NW * 4
CH_PLAN = (22, 22, 6, 6, 4)

# ------------------------------------------------------------
def quant_w_int8_per_layer_linear(w: torch.Tensor):
    """
    Quantização int8 por layer:
      scaleW = maxabs/127
      wq = round(w/scaleW) clamped
    Retorna (wq int8 numpy, scaleW float)
    """
    w_cpu = w.detach().cpu().float()
    maxabs = float(w_cpu.abs().max().item())
    if maxabs < 1e-12:
        scale = 1.0
        wq = torch.zeros_like(w_cpu, dtype=torch.int8)
    else:
        scale = maxabs / 127.0
        wq = torch.round(w_cpu / scale).clamp(-127, 127).to(torch.int8)
    return wq.numpy(), scale

def _pow2_q15_ge(q):
    """
    Retorna a menor potência de 2 >= q (q positivo), limitada ao range Q15.
    Usar >= evita reduzir demais o scale e saturar mais pesos.
    """
    if q <= 1:
        return 1
    # próxima potência de 2
    p = 1 << (q - 1).bit_length()
    if p > 32767:
        # maior potência de 2 representável em Q15 int16 é 16384 (2^14)
        p = 16384
    return p

def quant_w_int8_per_layer(w: torch.Tensor):
    """
    Quantização int8 por layer com scale quantizado em Q15 potência de 2:
      scale_real = maxabs/127
      scale_q15  = potência de 2 (Q15)
      scale      = scale_q15 / 32768
      wq = round(w/scale) clamped

    Retorna (wq int8 numpy, scale float, scale_q15 int, shift int)
    Onde:
      scale = scale_q15 / 32768
      scale_q15 = 2^k
      shift = 15 - k
    """
    w_cpu = w.detach().cpu().float()
    maxabs = float(w_cpu.abs().max().item())

    if maxabs < 1e-12:
        scale = 1.0
        scale_q15 = 32767   # caso especial só para consistência; wq será zero
        shift = 0
        wq = torch.zeros_like(w_cpu, dtype=torch.int8)
    else:
        # scale "ideal"
        scale_real = maxabs / 127.0

        # converte para Q15
        q15 = q15_from_float(scale_real)

        # garante positivo mínimo
        if q15 < 1:
            q15 = 1

        # força potência de 2 (em inteiro Q15)
        scale_q15 = _pow2_q15_ge(q15)

        # volta para float (esse é o scale realmente usado na quantização)
        scale = scale_q15 / 32768.0

        # quantiza pesos com esse scale "power-of-two Q15"
        wq = torch.round(w_cpu / scale).clamp(-127, 127).to(torch.int8)

        # shift equivalente (scale_q15 = 2^k)
        k = int(math.log2(scale_q15))
        shift = 15 - k

    return wq.numpy(), float(scale), int(scale_q15), int(shift)


def bias_float(b: torch.Tensor, cout: int):
    """
    Bias float verdadeiro do PyTorch.
    Se não existir, retorna zeros float.
    """
    if b is None:
        return np.zeros((cout,), dtype=np.float32)
    return b.detach().cpu().float().numpy().astype(np.float32)

def write_array_int8(f, name: str, arr: np.ndarray):
    flat = arr.flatten()
    f.write(f"#ifdef USE_{name}_w\n")
    f.write(f"static const int8_t {name}[{flat.size}] = {{\n  ")
    for i, v in enumerate(flat):
        f.write(f"{int(v)}, ")
        if (i + 1) % 16 == 0:
            f.write("\n  ")
    f.write("\n};\n\n")
    f.write(f"#endif // USE_{name}_w\n\n")


def fx16_from_float(x):
#     static inline fx16_t fx16_sat_i32(int32_t x) {
#   if (x > 32767) return 32767;
#   if (x < -32768) return -32768;
#   return (fx16_t)x;
# }

# // float -> fixed(Q8.8)
# static inline fx16_t fx16_from_float(float f) {
#   int ipart =  (int)f;
#   int frac = (f - (float)ipart) * 256.0f; // parte fracionária em 8 bits
#   int32_t v = (ipart << FX_FRAC_BITS) +frac;
#   return fx16_sat_i32(v);
# }
    ipart = int(math.floor(x))
    frac = int(round((x - ipart) * 256.0))
    v = (ipart << 8) + frac
    if v > 32767:
        v = 32767
    if v < -32768:
        v = -32768
    return v


def write_array_f32(f, name: str, arr: np.ndarray):
    flat = arr.flatten().astype(np.float32)
    f.write(f"static const float {name}[{flat.size}] = {{\n  ")
    for i, v in enumerate(flat):
        fv = float(v)

        # garante sintaxe float válida em C: "0.0f", "1.0f", "3.14f"
        # e também lida com notação científica
        s = f"{fv:.9g}"
        if ("e" not in s) and ("E" not in s) and ("." not in s):
            s += ".0"

        f.write(f"{s}f, ")
        if (i + 1) % 8 == 0:
            f.write("\n  ")
    f.write("\n};\n\n")

def f16_from_float(x):
    # float -> fixed(Q8.8)
    ipart = int(math.floor(x))
    frac = int(round((x - ipart) * 256.0))
    v = (ipart << 8) + frac
    if v > 32767:
        v = 32767
    if v < -32768:
        v = -32768
    return v

def write_array_f16(f, name: str, arr: np.ndarray):
    flat = arr.flatten().astype(np.float32)
    f.write(f"static const int16_t {name}[{flat.size}] = {{\n  ")
    for i, v in enumerate(flat):
        fv = float(v)
        f.write(f"{f16_from_float(fv)}, ")
        if (i + 1) % 8 == 0:
            f.write("\n  ")
    f.write("\n};\n\n")


def q15_from_float(x):
    q = int(x * 32768.0 + (0.5 if x >= 0.0 else -0.5))
    if q > 32767:
        q = 32767
    if q < -32768:
        q = -32768
    return q



def main():
    G = GeneratorMono24_Simple(nz=NZ, nw=NW, ch_plan=CH_PLAN)
    sd = torch.load(MODEL_PATH, map_location="cpu")
    # Remove bias keys if the model does not use bias in some layers
    for k in list(sd.keys()):
        if k.endswith('.bias') and k not in G.state_dict():
            del sd[k]
    G.load_state_dict(sd, strict=True)
    G.eval()

    ztest =   torch.randn(64, NZ, 1, 1, device="cpu")

    applyCorrection = False 
    with torch.no_grad():
        #get the value from each layer
        x = 16*ztest.view(64, NW, 2, 2)
        x0 = G.l0(x)
        print(f"l0 min/max = {x0.min().item():.6f}/{x0.max().item():.6f}")

        x0u = torch.nn.functional.interpolate(x0, size=(6, 6), mode="nearest")
        x1 = G.l1(x0u)
        while applyCorrection and x1.max().item() > 64.0: #if the output of a layer is greater than 64, we need to apply a scale correction to the next layer input
              #scale G.l1.conv weights and bias
              scale = 64.0 / x1.max().item()
              G.l1.conv.weight.data *= scale
              if G.l1.conv.bias is not None:
                  G.l1.conv.bias.data *= scale
              print(f"Applied scale correction to l1: {scale:.6f}")
              x1 = G.l1(x0u)
              
        print(f"l1 min/max = {x1.min().item():.6f}/{x1.max().item():.6f}")

        x1u = torch.nn.functional.interpolate(x1, size=(12, 12), mode="nearest")
        x2 = G.l2(x1u)
        while applyCorrection and x2.max().item() > 64.0: #if the output of a layer is greater than 64, we need to apply a scale correction to the next layer input
              #scale G.l2.conv weights and bias
              scale = 64.0 / x2.max().item()
              G.l2.conv.weight.data *= scale
              if G.l2.conv.bias is not None:
                  G.l2.conv.bias.data *= scale
              print(f"Applied scale correction to l2: {scale:.6f}")
              x2 = G.l2(x1u)
        print(f"l2 min/max = {x2.min().item():.6f}/{x2.max().item():.6f}")

        x2u = torch.nn.functional.interpolate(x2, size=(24, 24), mode="nearest")
        x3 = G.l3(x2u)
        while applyCorrection and x3.max().item() > 64.0: #if the output of a layer is greater than 64, we need to apply a scale correction to the next layer input
              #scale G.l3.conv weights and bias
              scale = 64.0 / x3.max().item()
              G.l3.conv.weight.data *= scale
              if G.l3.conv.bias is not None:
                  G.l3.conv.bias.data *= scale
              print(f"Applied scale correction to l3: {scale:.6f}")
              x3 = G.l3(x2u)
        print(f"l3 min/max = {x3.min().item():.6f}/{x3.max().item():.6f}")

        x4 = G.l4(x3)
        while applyCorrection and x4.max().item() > 64.0: #if the output of a layer is greater than 64, we need to apply a scale correction to the next layer input
              #scale G.l4.conv weights and bias
              scale = 64.0 / x4.max().item()
              G.l4.conv.weight.data *= scale
              if G.l4.conv.bias is not None:
                  G.l4.conv.bias.data *= scale
              print(f"Applied scale correction to l4: {scale:.6f}")
              x4 = G.l4(x3)
        print(f"l4 min/max = {x4.min().item():.6f}/{x4.max().item():.6f}")

        #out min/max = -9.336440/226.111389
        xout = G.out(x4)
        while applyCorrection and xout.max().item() > 120.0: #if the output of a layer is greater than 64, we need to apply a scale correction to the next layer input
              #scale G.out.conv weights and bias
              scale = 120.0 / xout.max().item()
              G.out.conv.weight.data *= scale
              if G.out.conv.bias is not None:
                  G.out.conv.bias.data *= scale
              print(f"Applied scale correction to out: {scale:.6f}")
              xout = G.out(x4)
        print(f"out min/max = {xout.min().item():.6f}/{xout.max().item():.6f}")

        x = G.tanh(xout/16) #ajuste conforme seu modelo (se usar tanh, e qual escala pré-tanh)

 


 

    # convs (ajuste se seus nomes forem diferentes)
    convs = [
        ("l0",  G.l0.conv),
        ("l1",  G.l1.conv),
        ("l2",  G.l2.conv),
        ("l3",  G.l3.conv),
        ("l4",  G.l4.conv),
        ("out", G.out.conv),
    ]

    # quantiza e coleta
    WQ = {}
    SCALEW = {}
    BF = {}

    for name, conv in convs:
        wq, sw  = quant_w_int8_per_layer_linear(conv.weight.data)
        
        
 
        WQ[name] = wq
        SCALEW[name] = float(sw)

        cout = wq.shape[0]
        BF[name] = bias_float(conv.bias.data if conv.bias is not None else None, cout)

        nz_cnt = int(np.count_nonzero(wq))
        print(f"{name}: wq nonzero {nz_cnt}/{wq.size} max|wq|={int(np.max(np.abs(wq)))} scaleW={sw:g}")

    # escreve header
    c2, c6, c12, c24, cH = CH_PLAN
    with open(OUT_H, "w", encoding="utf-8") as f:
        f.write("// Auto-generated weights for FLOAT runtime\n")
        f.write("#pragma once\n\n")
        f.write("#include <stdint.h>\n\n")

        f.write(f"#define G_NW   {NW}\n")
        f.write(f"#define G_NZ   {NZ}\n")
        f.write(f"#define G_C2   {c2}\n")
        f.write(f"#define G_C6   {c6}\n")
        f.write(f"#define G_C12  {c12}\n")
        f.write(f"#define G_C24  {c24}\n")
        f.write(f"#define G_CH   {cH}\n\n")

        # escalas (float) — esses nomes batem com o gen_runtime_float.c que te mandei
        f.write("// Weight dequant scales (float): w_float = w_q * SCALEW_*\n")
        f.write(f"static const float SCALEW_L0  = {SCALEW['l0']:.10g}f;\n")
        f.write(f"static const float SCALEW_L1  = {SCALEW['l1']:.10g}f;\n")
        f.write(f"static const float SCALEW_L2  = {SCALEW['l2']:.10g}f;\n")
        f.write(f"static const float SCALEW_L3  = {SCALEW['l3']:.10g}f;\n")
        f.write(f"static const float SCALEW_L4  = {SCALEW['l4']:.10g}f;\n")
        f.write(f"static const float SCALEW_OUT = {SCALEW['out']:.10g}f;\n\n")

        f.write("// Weight dequant as Q15 \n")
        f.write("// w_q15 = round(w_float/scaleW * 32768) clamped\n")
        f.write(f"static const  int16_t SCALEW_L0_Q15  = {   q15_from_float(SCALEW['l0'])};\n")
        f.write(f"static const  int16_t SCALEW_L1_Q15  = {   q15_from_float(SCALEW['l1'])};\n")
        f.write(f"static const  int16_t SCALEW_L2_Q15  = {   q15_from_float(SCALEW['l2'])};\n")
        f.write(f"static const  int16_t SCALEW_L3_Q15  = {   q15_from_float(SCALEW['l3'])};\n")
        f.write(f"static const  int16_t SCALEW_L4_Q15  = {   q15_from_float(SCALEW['l4'])};\n")
        f.write(f"static const  int16_t SCALEW_OUT_Q15 = {   q15_from_float(SCALEW['out'])};\n\n")



        # pesos int8
        
        write_array_int8(f, "W_l0",  WQ["l0"])
        write_array_int8(f, "W_l1",  WQ["l1"])
        write_array_int8(f, "W_l2",  WQ["l2"])
        write_array_int8(f, "W_l3",  WQ["l3"])
        write_array_int8(f, "W_l4",  WQ["l4"])
        write_array_int8(f, "W_out", WQ["out"])

        # bias float verdadeiro
        f.write("// Bias float (original)\n")
        write_array_f32(f, "B_l0",  BF["l0"])
        write_array_f32(f, "B_l1",  BF["l1"])
        write_array_f32(f, "B_l2",  BF["l2"])
        write_array_f32(f, "B_l3",  BF["l3"])
        write_array_f32(f, "B_l4",  BF["l4"])
        write_array_f32(f, "B_out", BF["out"])

        # bias float verdadeiro
        f.write("// Bias float (original)\n")
        write_array_f16(f, "B_l0_f16",  BF["l0"])
        write_array_f16(f, "B_l1_f16",  BF["l1"])
        write_array_f16(f, "B_l2_f16",  BF["l2"])
        write_array_f16(f, "B_l3_f16",  BF["l3"])
        write_array_f16(f, "B_l4_f16",  BF["l4"])
        write_array_f16(f, "B_out_f16", BF["out"])


    print(f"OK: wrote {OUT_H}")

if __name__ == "__main__":
    main()
