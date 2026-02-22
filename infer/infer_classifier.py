#!/usr/bin/env python3
"""
Script de inferência para predizer tags/atributos de uma imagem.
Carrega modelo treinado e retorna valores contínuos [-1, 1] para cada atributo.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Try to reuse the same downscale transform used during training
try:
    from train_classifier import DownscaleTransform
except Exception:
    DownscaleTransform = None

# ============================================================================
# Definição do modelo (mesma arquitetura do treino)
# ============================================================================

FIELD_DIMS = {
    'gender': 1,
    'skin_tone': 1,
    'hair_type': 1,
    'hair_tone': 1,
    'style_length': 1,
    'style_type': 1,
}


class MultiTaskRegressor(nn.Module):
    def __init__(self, nc=1, nf=32, image_size=24):
        super().__init__()
        self.image_size = image_size

        self.conv1 = nn.Conv2d(nc, nf, 4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)

        self.conv2 = nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(nf * 2)

        self.conv3 = nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(nf * 4)

        if image_size == 24:
            self.conv4 = nn.Conv2d(nf * 4, nf * 8, 3, 1, 0, bias=False)
            feature_dim = nf * 8
        elif image_size == 32:
            self.conv4 = nn.Conv2d(nf * 4, nf * 8, 4, 1, 0, bias=False)
            feature_dim = nf * 8
        else:
            raise ValueError(f"image_size={image_size} não suportado. Use 24 ou 32.")

        self.bn4 = nn.BatchNorm2d(nf * 8)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.5)

        self.heads = nn.ModuleDict()
        for field, dim in FIELD_DIMS.items():
            self.heads[field] = nn.Sequential(
                nn.Linear(feature_dim, dim),
                nn.Tanh()
            )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.leaky_relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.leaky_relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.leaky_relu(x)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        outputs = {
            field: self.heads[field](x)
            for field in FIELD_DIMS.keys()
        }

        return outputs


# ============================================================================
# Mapeamentos inversos: valor -> label mais próximo
# ============================================================================

# Gênero: -1 = mulher, +1 = homem
def value_to_gender(val):
    if val < 0:
        return 'woman', abs(val)
    else:
        return 'man', abs(val)


# Cor de pele: -1 = muito claro, +1 = muito escuro
# Use the same mapping used during training (train_classifier.COR_PELE_TO_VALUE)
# Mapping: -1 = very_light, +1 = very_dark with intermediate steps
# Use the canonical four skin-tone labels used by the training pipeline
COR_PELE_VALUES = [
    (-1.0, 'light'),
    (-0.3333, 'medium'),
    (0.3333, 'dark'),
    (1.0, 'very_dark'),
]

def value_to_skin_tone(val):
    """Map continuous model output to the closest training label and a confidence.

    Confidence is derived from distance to the nearest target value, scaled
    so that exact match -> 1.0 and maximum possible distance (2.0) -> 0.0.
    """
    min_dist = float('inf')
    best_label = 'medium'
    best_target = 0.0
    for target_val, label in COR_PELE_VALUES:
        dist = abs(val - target_val)
        if dist < min_dist:
            min_dist = dist
            best_label = label
            best_target = target_val
    # distance range is [0, 2] across the [-1,1] interval; normalize to [0,1]
    confidence = max(0.0, 1.0 - (min_dist / 2.0))
    return best_label, confidence


# Tipo de cabelo: -1 = liso, +1 = crespo
TIPO_CABELO_VALUES = [
    (-1.0, 'straight'),
    (-0.3333, 'wavy'),
    (0.3333, 'curly'),
    (0.0, 'bald'),
]

def value_to_hair_type(val):
    min_dist = float('inf')
    best_label = 'curly'
    for target_val, label in TIPO_CABELO_VALUES:
        dist = abs(val - target_val)
        if dist < min_dist:
            min_dist = dist
            best_label = label
    confidence = max(0.0, 1.0 - (min_dist / 2.0))
    return best_label, confidence


# Cor de cabelo: -1 = escuro, +1 = claro
# Map hair color to coarse labels used by the classifier API
# Use representative points derived from train_classifier.COR_CABELO_TO_VALUE
# Use the exact training target points from train_classifier.COR_CABELO_TO_VALUE
COR_CABELO_VALUES = [
    (-1.0, 'black'),
    (-0.3333, 'brown_dark'),
    (0.3333, 'brown_light'),
    (1.0, 'light'),
]

def value_to_hair_tone(val):
    min_dist = float('inf')
    best_label = 'brown'
    for target_val, label in COR_CABELO_VALUES:
        dist = abs(val - target_val)
        if dist < min_dist:
            min_dist = dist
            best_label = label
    confidence = max(0.0, 1.0 - (min_dist / 2.0))
    return best_label, confidence


# Estilo comprimento: -1 = careca, +1 = muito longo
# Use coarse length targets only
ESTILO_COMPRIMENTO_VALUES = [
    (-1.0, 'bald'),
    (-0.2, 'short'),
    (0.2, 'medium'),
    (0.6, 'long'),
]

def value_to_style_length(val):
    # pick nearest coarse length label
    min_dist = float('inf')
    best_label = 'medium'
    for target_val, label in ESTILO_COMPRIMENTO_VALUES:
        dist = abs(val - target_val)
        if dist < min_dist:
            min_dist = dist
            best_label = label
    confidence = max(0.0, 1.0 - (min_dist / 2.0))
    return best_label, confidence


# Estilo tipo: -1 = simples, +1 = elaborado (coarse)
ESTILO_TIPO_VALUES = [
    (0.0, 'bald'),
    (-0.1, 'short'),
    (0.0, 'medium'),
    (0.1, 'long'),
]

def value_to_style_type(val):
    min_dist = float('inf')
    best_label = 'medium'
    for target_val, label in ESTILO_TIPO_VALUES:
        dist = abs(val - target_val)
        if dist < min_dist:
            min_dist = dist
            best_label = label
    confidence = max(0.0, 1.0 - (min_dist / 2.0))
    return best_label, confidence


def values_to_hair_style(comprimento_val, tipo_val):
    """Determine final hair style (coarse) based on length only."""
    comp_label, comp_conf = value_to_style_length(comprimento_val)
    return comp_label, comp_conf


# ============================================================================
# Funções de inferência
# ============================================================================

def load_model(checkpoint_path, image_size=24, nf=32, device='cuda'):
    """Carrega o modelo treinado."""
    model = MultiTaskRegressor(nc=1, nf=nf, image_size=image_size).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def preprocess_image(image_path, image_size=24):
    """Carrega e prepara imagem para inferência."""
    image = Image.open(image_path).convert('L')

    # If the image is larger than the target size, prefer the DownscaleTransform
    # used during training to match preprocessing. Otherwise use the fallback.
    w, h = image.size
    if (w > image_size or h > image_size):
        if DownscaleTransform is not None:
            down = DownscaleTransform(image_size=image_size, normalize=True)
            t = down(image)  # returns tensor [1,H,W] or [B,1,H,W]
            if t.dim() == 3:
                return t.unsqueeze(0)  # -> [B,1,H,W]
            elif t.dim() == 4:
                return t
        else:
            print(f"Aviso: DownscaleTransform não disponível; usando fallback para imagem {image_path}")

    # Fallback to simple torchvision pipeline if DownscaleTransform not used
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    return transform(image).unsqueeze(0)


def predict_tags(model, image_tensor, device='cuda'):
    """Realiza predição e retorna valores contínuos e labels."""
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)

    # Extrair valores
    values = {
        field: outputs[field].cpu().item()
        for field in FIELD_DIMS.keys()
    }

    # Convert to labels using the training mappings
    gender_label, gender_conf = value_to_gender(values['gender'])
    skin_label, skin_conf = value_to_skin_tone(values['skin_tone'])
    hair_type_label, hair_type_conf = value_to_hair_type(values['hair_type'])

    # If predicted hair type is bald, set a default hair_tone
    if hair_type_label == 'bald':
        hair_tone_label = 'black'
        hair_tone_conf = 0.35
    else:
        hair_tone_label, hair_tone_conf = value_to_hair_tone(values['hair_tone'])

    style_label, style_conf = values_to_hair_style(
        values['style_length'],
        values['style_type']
    )

    # Build results with English keys matching the training pipeline
    results = {
        'values': {
            'gender': values.get('gender'),
            'skin_tone': values.get('skin_tone'),
            'hair_type': values.get('hair_type'),
            'hair_tone': values.get('hair_tone'),
            'hair_style': values.get('style_length'),
        },
        'labels': {
            'gender': gender_label,
            'skin_tone': skin_label,
            'hair_type': hair_type_label,
            'hair_tone': hair_tone_label,
            'hair_style': style_label,
        },
        'confidence': {
            'gender': gender_conf,
            'skin_tone': skin_conf,
            'hair_type': hair_type_conf,
            'hair_tone': hair_tone_conf,
            'hair_style': style_conf,
        }
    }

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Inferência de tags/atributos de pessoa em uma imagem'
    )
    parser.add_argument('image', help='Caminho para a imagem')
    parser.add_argument(
        '--checkpoint',
        default='output_classifier/regressor_best.pth',
        help='Caminho para o checkpoint do modelo'
    )
    parser.add_argument('--image-size', type=int, default=24, choices=[24, 32])
    parser.add_argument('--nf', type=int, default=32)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--show-values', action='store_true',
                       help='Mostrar valores contínuos [-1, 1]')
    parser.add_argument('--show-confidence', action='store_true',
                       help='Mostrar confiança das predições')

    args = parser.parse_args()

    # Verificar se imagem existe
    if not Path(args.image).exists():
        print(f"Erro: imagem não encontrada: {args.image}")
        sys.exit(1)

    # Verificar se checkpoint existe
    if not Path(args.checkpoint).exists():
        print(f"Erro: checkpoint não encontrado: {args.checkpoint}")
        sys.exit(1)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Carregar modelo
    print(f"Carregando modelo de {args.checkpoint}...")
    model = load_model(args.checkpoint, args.image_size, args.nf, device)

    # Carregar e processar imagem
    print(f"Processando imagem: {args.image}")
    image_tensor = preprocess_image(args.image, args.image_size)

    # Predição
    results = predict_tags(model, image_tensor, device)

    # Exibir resultados
    print("\n" + "="*60)
    print("TAGS PREDITAS")
    print("="*60)

    for field, label in results['labels'].items():
        output = f"{field:20s}: {label}"

        if args.show_confidence:
            conf = results['confidence'][field]
            output += f" (conf: {conf:.2f})"

        if args.show_values:
            val = results['values'][field]
            output += f" [valor: {val:+.3f}]"

        print(output)

    print("="*60)

    # Se quiser valores raw
    if args.show_values:
        print("\nValores contínuos (range [-1, 1]):")
        for field, val in results['values'].items():
            print(f"  {field:20s}: {val:+.4f}")


if __name__ == '__main__':
    main()
