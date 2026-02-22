import torch
import torch.nn.functional as F

def unsharp_mask(x, radius=1.0, amount=0.8, eps=1e-6):
    """
    x: Tensor [B,1,H,W] ou [1,H,W] em [0,1]
    radius: controla o blur (sigma)
    amount: força da nitidez
    """
    if x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.dim() == 4 and x.size(1) == 1

    # blur gaussiano separável
    sigma = radius
    k = int(2 * round(3 * sigma) + 1)
    k = max(k, 3)
    if k % 2 == 0:
        k += 1

    # kernel 1D gaussiano
    t = torch.arange(k, device=x.device, dtype=x.dtype) - (k - 1) / 2
    g = torch.exp(-(t * t) / (2 * sigma * sigma + eps))
    g = g / (g.sum() + eps)

    gH = g.view(1, 1, 1, k)
    gV = g.view(1, 1, k, 1)

    # padding "reflect" evita borda escura
    pad = k // 2
    y = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y = F.conv2d(y, gH)
    y = F.conv2d(y, gV)

    # unsharp: x + amount*(x - blur)
    sharp = x + amount * (x - y)
    return sharp.clamp(0.0, 1.0)

def downscale_portrait_24x24(gray, size=24,
                            pre_blur=None,
                            pre_sharpen=(1.0, 0.8),
                            post_sharpen=(0.5, 0.4),
                            mode="bicubic"):
    """
    gray: [H,W] ou [1,H,W] ou [B,1,H,W] em [0,1]
    mode: "area" (melhor pra reduzir) ou "bicubic"
    """
    x = gray
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(0)
    assert x.dim() == 4 and x.size(1) == 1

    # (opcional) um blur leve ANTES pode reduzir aliasing
    if pre_blur is not None:
        x = unsharp_mask(x, radius=pre_blur, amount=-0.0)  # só blur se quiser (ver nota abaixo)

    # sharpen leve antes do resize ajuda detalhes (olhos/sobrancelhas)
    if pre_sharpen is not None:
        r, a = pre_sharpen
        x = unsharp_mask(x, radius=r, amount=a)

    # resize (AREA é excelente pra downscale; bicubic também ok)
    x = F.interpolate(x, size=(size, size), mode=mode, align_corners=False if mode in ["bilinear","bicubic"] else None)

    # sharpen leve depois pra recuperar micro-contraste
    if post_sharpen is not None:
        r, a = post_sharpen
        x = unsharp_mask(x, radius=r, amount=a)

    return x.squeeze(0).squeeze(0)  # [24,24]
