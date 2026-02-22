from __future__ import print_function

import argparse
import os
import random
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from PIL import ImageEnhance

import torch.nn.functional as F

from model import (
    GeneratorMono24_Simple,
    CriticMono24_Simple,
    weights_init,
    adjust_layer_amplitude_sequential

)

# downscale helper (local)
from downscale import downscale_portrait_24x24



class DownscaleTransform:
    """Apply `downscale_portrait_24x24` to a tensor image.

    Expects input from `transforms.ToTensor()` (shape [1,H,W], range [0,1]).
    Returns a tensor [1,size,size] in [0,1].
    """
    def __init__(self, size=24, pre_blur=None, pre_sharpen=(1.0, 0.3), post_sharpen=(0.5, 0.3), mode="bicubic"):
        self.size = size
        self.pre_blur = pre_blur
        self.pre_sharpen = pre_sharpen
        self.post_sharpen = post_sharpen
        self.mode = mode

    def __call__(self, tensor):
        # tensor: [C,H,W] with C==1
        if tensor.dim() == 3 and tensor.size(0) == 1:
            t = tensor
        elif tensor.dim() == 2:
            t = tensor.unsqueeze(0)
        else:
            t = tensor

        out = downscale_portrait_24x24(
            t, size=self.size, pre_blur=self.pre_blur,
            pre_sharpen=self.pre_sharpen, post_sharpen=self.post_sharpen,
            mode=self.mode
        )

        # ensure [1,H,W]
        if out.dim() == 2:
            out = out.unsqueeze(0)
        return out


class ContrastEnhancement:
    def __init__(self, factor=1.3):
        self.factor = factor

    def __call__(self, img):
        enhancer = ImageEnhance.Contrast(img)
        return enhancer.enhance(self.factor)


# ============================================================================
# WGAN-GP: Gradient Penalty
# ============================================================================
def gradient_penalty(critic, real_data, fake_data, device):
    """
    Calcula gradient penalty (WGAN-GP) para estabilizar o treinamento.
    """
    batch_size = real_data.size(0)

    # Interpolação aleatória entre real e fake
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_data + (1 - alpha) * fake_data).requires_grad_(True)

    # Critic scores para interpolados
    critic_interpolates = critic(interpolates)

    # Gradientes em relação aos interpolados
    gradients = torch.autograd.grad(
        outputs=critic_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Calcula norma L2 dos gradientes
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)

    # Penalty: (||grad|| - 1)^2
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


# ==========================================================================
# Diversity repulsion (cosine) - anti-mode-collapse
# ==========================================================================
def diversity_repulsion_loss(fake, eps=1e-8):
    # fake: [B,1,H,W] em [-1,1]
    B = fake.size(0)
    x = fake.view(B, -1)
    x = x - x.mean(dim=1, keepdim=True)
    x = F.normalize(x, dim=1, eps=eps)
    sim = x @ x.t()                 # [B,B], diag = 1
    sim = sim - torch.eye(B, device=fake.device)
    return sim.pow(2).mean()


# ============================================================================
# Argumentos
# ============================================================================

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='folder', help='imagenet | folder | lfw')
parser.add_argument('--dataroot', default='images_v6', help='path to dataset')
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--batchSize', type=int, default=256)
parser.add_argument('--imageSize', type=int, default=24)

parser.add_argument('--nz', type=int, default=64, help='latent z size (must satisfy nz == nw*4 for 2x2 reshape)')
parser.add_argument('--nw', type=int, default=16, help='channels after reshape: NZ == NW*4 (ex: 64->16*4)')
parser.add_argument('--ngf', type=int, default=9)
parser.add_argument('--ndf', type=int, default=9)
parser.add_argument('--n_bits', type=int, choices=[2, 4], default=4, help='Número de bits para quantização (2 ou 4)')

parser.add_argument('--niter', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.0003, help='learning rate (WGAN geralmente usa LR menor)')
parser.add_argument('--beta1', type=float, default=0.0, help='beta1 para Adam (WGAN-GP: 0.0 ou 0.5)')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 para Adam')
parser.add_argument('--dry-run', action='store_true')
parser.add_argument('--ngpu', type=int, default=1)

parser.add_argument('--netG', default='')
parser.add_argument('--netD', default='')
parser.add_argument('--outf', default='output_wgan_4bit_24')
parser.add_argument('--manualSeed', type=int, default=None)

# WGAN-GP específico
parser.add_argument('--lambda_gp', type=float, default=10.0, help='gradient penalty coefficient')
parser.add_argument('--n_critic', type=int, default=5, help='treina critic N vezes por cada G')

# Estabilização
parser.add_argument('--warmupG', type=int, default=0, help='epochs: treina só G no começo (D congelado)')
parser.add_argument('--lrD_mult', type=float, default=1.2, help='multiplicador LR do Critic')
parser.add_argument('--lrG_mult', type=float, default=1.0, help='multiplicador LR do G')

# Diversity repulsion (G)
parser.add_argument('--lambda_div', type=float, default=0.0001)

opt = parser.parse_args()
print(opt)

os.makedirs(opt.outf, exist_ok=True)

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed:", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    if opt.ngpu > torch.cuda.device_count():
        opt.ngpu = torch.cuda.device_count()
else:
    device = torch.device("cpu")
    opt.ngpu = 0

print(f"Using device: {device} | GPUs: {opt.ngpu}")

if opt.imageSize != 24:
    raise ValueError(f"Este script está configurado para 24x24. Recebido imageSize={opt.imageSize}.")
if opt.nz != opt.nw * 4:
    raise ValueError(f"Para reshape 2x2 precisa nz == nw*4. Recebido nz={opt.nz}, nw={opt.nw}.")


# ============================================================================
# Dataset
# ============================================================================


dataset = dset.ImageFolder(
    root=opt.dataroot,
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((2*opt.imageSize, 2*opt.imageSize)),  # pré-redimensionamento para evitar artefatos extremos
        ContrastEnhancement(factor=1.3),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.clamp(x ** 0.8, 0.0, 1.0)),
        DownscaleTransform(size=opt.imageSize, pre_blur=None ),
        transforms.Normalize((0.5,), (0.5,)),
    ])
)
nc = 1

# Garante que o dataset seja múltiplo do batch size
num_imgs = len(dataset)
bs = opt.batchSize
num_keep = (num_imgs // bs) * bs
if num_keep < num_imgs:
    print(f"[info] Dataset original: {num_imgs} imagens. Mantendo {num_keep} para múltiplo exato do batch size ({bs}).")
    dataset.samples = dataset.samples[:num_keep]
    if hasattr(dataset, 'imgs'):
        dataset.imgs = dataset.imgs[:num_keep]

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=opt.batchSize, shuffle=True,
    num_workers=int(opt.workers), pin_memory=True
)

ngpu = int(opt.ngpu)
nz = int(opt.nz)
nw = int(opt.nw)
ngf = int(opt.ngf)
ndf = int(opt.ndf)


# ============================================================================
# Redes
# ============================================================================


netG = GeneratorMono24_Simple(  nz=nz, nw=nw ).to(device)
netG.apply(weights_init)

# Discriminator sem sigmoid (WGAN precisa de saída linear)
netD = CriticMono24_Simple(   ).to(device)
netD.apply(weights_init)

if ngpu > 1:
    netG = nn.DataParallel(netG, device_ids=list(range(ngpu)))
    netD = nn.DataParallel(netD, device_ids=list(range(ngpu)))

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG, map_location=device))
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD, map_location=device))

print(netG)
print(netD)

#print the number of parameters
num_params_G = sum(p.numel() for p in netG.parameters())
num_params_D = sum(p.numel() for p in netD.parameters())
print(f"Number of parameters - Generator: {num_params_G}, Discriminator: {num_params_D}")

# ============================================================================
# Optimizers (sem loss function, WGAN usa Wasserstein distance)
# ============================================================================

lambda_div = opt.lambda_div
lambda_gp = opt.lambda_gp

fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

# WGAN-GP: RMSProp ou Adam com beta1=0.0, beta2=0.9
optimizerD = optim.Adam(
    netD.parameters(),
    lr=opt.lr * opt.lrD_mult,
    betas=(opt.beta1, opt.beta2)
)
optimizerG = optim.Adam(
    netG.parameters(),
    lr=opt.lr * opt.lrG_mult,
    betas=(opt.beta1, opt.beta2)
)

schedulerD = optim.lr_scheduler.StepLR(optimizerD, step_size=50, gamma=0.95)
schedulerG = optim.lr_scheduler.StepLR(optimizerG, step_size=50, gamma=0.95)

if opt.dry_run:
    opt.niter = 1

# =========================================================================
# Normalization Loss 
# =========================================================================

def range_penalty_tensor(x, limit=127.0):
    overflow = F.relu(x.abs() - limit)
    return (overflow ** 2).mean()

def range_penalty_model(model, limit=127.0):
    loss = 0.0
    for name, act in model.activations.items():
        loss = loss + range_penalty_tensor(act, limit=limit)
    return loss


# ============================================================================
# Treino WGAN-GP
# ============================================================================

print("\n🚀 Iniciando treinamento WGAN-GP")
print(f"   Critic iterations: {opt.n_critic} | GP lambda: {opt.lambda_gp}")

half_epoch = len(dataloader) // 2

for epoch in range(opt.niter):

    # quant schedule (conservador: entra mais tarde)
    if epoch < opt.niter * 0.05:
        quant_temp = 0.1
    else:
        progress = (epoch - opt.niter * 0.05) / (opt.niter * 0.7)
        quant_temp = float(np.clip(0.1 + progress * 0.4, 0.1, 0.50))

    epoch_losses_D = []
    epoch_losses_G = []
    epoch_wasserstein = []

    for i, data in enumerate(dataloader, 0):

        real_cpu = data[0].to(device)
        batch_size = real_cpu.size(0)

        # ----------------------------
        # (1) Update Critic (n_critic vezes)
        # ----------------------------
        do_critic = (epoch >= opt.warmupG)

        if do_critic:
            for _ in range(opt.n_critic):
                netD.zero_grad()

                # Score real (sem sigmoid, saída linear)
                real_score = netD(real_cpu)

                # Score fake (detach)
                z = torch.randn(batch_size, nz, 1, 1, device=device)
                fake = netG(z, quant_temp).detach()
                fake_score = netD(fake)

                # Wasserstein loss: maximize E[D(real)] - E[D(fake)]
                # Para minimizar: -E[D(real)] + E[D(fake)]
                wasserstein_d = real_score.mean() - fake_score.mean()

                # Gradient penalty
                gp = gradient_penalty(netD, real_cpu, fake, device)

                # Loss total do Critic
                errD = -wasserstein_d + lambda_gp * gp   
                errD.backward()
                optimizerD.step()

            # Métricas (última iteração)
            D_real = real_score.mean().item()
            D_fake = fake_score.mean().item()
            wasserstein_distance = wasserstein_d.item()
        else:
            D_real = 0.0
            D_fake = 0.0
            wasserstein_distance = 0.0
            errD = torch.tensor(0.0, device=device)

        # ----------------------------
        # (2) Update Generator
        # ----------------------------
        netG.zero_grad()

        z = torch.randn(batch_size, nz, 1, 1, device=device)
        fake = netG(z, quant_temp)

        # G quer maximizar D(fake) => minimizar -D(fake)
        fake_score_g = netD(fake)
        errG_wgan = -fake_score_g.mean()

        # Diversity loss (opcional)
        errG = errG_wgan + lambda_div * diversity_repulsion_loss(fake)  
        errG.backward()
        optimizerG.step()

        D_G_z = fake_score_g.mean().item()

        # Tracking
        epoch_losses_D.append(float(errD.item()) if hasattr(errD, "item") else 0.0)
        epoch_losses_G.append(float(errG.item()))
        epoch_wasserstein.append(wasserstein_distance)

        status = ""
        if epoch < opt.warmupG:
            status = " (warmup: Critic congelado)"

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f W_dist: %.4f D(real): %.3f D(fake): %.3f D(G(z)): %.3f | Quant: %.1f%%%s'
            % (epoch, opt.niter, i, len(dataloader),
                float(errD.item()) if hasattr(errD, "item") else 0.0,
                float(errG.item()),
                wasserstein_distance,
                D_real, D_fake, D_G_z,
                quant_temp * 100, status))

        if (i % half_epoch == 0 ) and i > 0:
            print(f"Realimges range [{real_cpu.min().item():.3f}, {real_cpu.max().item():.3f}] | Fake images range [{fake.min().item():.3f}, {fake.max().item():.3f}]")
            vutils.save_image(real_cpu, '%s/real_samples.png' % opt.outf, normalize=True )
            fake_vis = netG(fixed_noise, quant_temp)
            vutils.save_image(
                fake_vis.detach(),
                '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                normalize=True 
            )
            vutils.save_image(
                fake_vis.detach(),
                '%s/fake_samples.png' % (opt.outf),
                normalize=True 
            )

        if opt.dry_run:
            break


        #if i % (half_epoch//4) == 0 and i > 0:
        #    adjust_layer_amplitude_sequential(netG,  torch.randn(opt.batchSize, nz, 1, 1, device=device) )

    schedulerD.step()
    schedulerG.step()

    avg_wasserstein = float(np.mean(epoch_wasserstein)) if epoch_wasserstein else 0.0

    force_save = (epoch < 3) or (epoch % 10 == 0)

    # Aplica o fold de BN nos pesos do gerador ANTES de salvar
    #adjust_layer_amplitude_sequential(netG,  torch.randn(opt.batchSize, nz, 1, 1, device=device))

    if ngpu > 1:
        g_module = netG.module
        d_state = netD.module.state_dict()
    else:
        g_module = netG
        d_state = netD.state_dict()

    g_fold = copy.deepcopy(g_module).to("cpu")
    g_fold.eval()
    g_state = g_fold.state_dict()

    if force_save:
        torch.save(g_state, '%s/netG_wgan_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(d_state, '%s/netD_wgan_epoch_%d.pth' % (opt.outf, epoch))
        print(f"✓ Checkpoint salvo: epoch {epoch} (quant_temp={quant_temp:.2f}) | W_dist={avg_wasserstein:.3f}")

print("\n✓ Treinamento WGAN-GP concluído!")
