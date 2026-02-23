# MSX Portrait Generator

A neural network-based portrait generator running on MSX, an 8-bit computer powered by the Z80 processor. This project implements a complete pipeline from training a Generative Adversarial Network (GAN) to deploying it as a ROM cartridge for real MSX hardware.

![MSX Portrait Generator Screenshots](images/openmsx0023.png)

## Overview

This project brings AI-generated portraits to the constraints of 1980s hardware. It generates 24x24 pixel monochrome portraits in real-time on the MSX, allowing users to customize physical characteristics or generate random portraits.

The generator uses a quantized Wasserstein GAN (WGAN) trained on portrait data, then converted to 8-bit integer operations that can run efficiently on the Z80 processor.

## Features

- **Customizable Portrait Generation**: Select specific physical characteristics:
  - Gender (Man/Woman)
  - Hair Style (Long/Medium/Short)
  - Hair Tone (Black/Dark Brown/Light Brown/Light)
  - Hair Type (Bald/Curly/Straight/Wavy)
  - Skin Tone (Dark/Medium)

- **Random Generation**: Generate completely random portraits with a single button press

- **Real Hardware Support**: Runs on actual MSX computers or emulators like openMSX

- **ROM Cartridge**: Packaged as a ROM file that can be loaded directly into MSX hardware

## Technical Details

### Architecture

- **Training Framework**: PyTorch-based WGAN with 4-bit weight quantization
- **Model**: Compact generator network optimized for 8-bit inference
- **Input**: 64-dimensional latent vector (z-space)
- **Output**: 24x24 pixel monochrome image
- **Runtime**: Pure C implementation with quantized integer operations
- **Target Hardware**: MSX (Z80 @ 3.58 MHz)

### Quantization

The network uses aggressive quantization to fit within MSX constraints:
- 4-bit weights for most layers
- 8-bit activations
- Integer-only arithmetic (no floating point)
- Optimized matrix operations for Z80

## Project Structure

```
.
├── train/                      # Neural network training code
│   ├── train_wgan_v4.py       # Main training script (WGAN)
│   ├── model.py               # Network architecture definitions
│   ├── downscale.py           # Image preprocessing utilities
│   └── output_wgan_4bit_24/   # Trained model checkpoints
│
├── runtime/                    # C runtime for testing
│   ├── main_gen.c             # Standalone test program
│   ├── gen_runtime.c          # Generator inference engine
│   ├── gen_weights.h          # Exported network weights
│   ├── model.py               # Export utilities
│   └── export_gl.py           # Weight export script
│
├── src/                        # MSX cartridge source code
│   ├── dgan.c                 # Main application code
│   ├── dgan_s*.c              # Stage-specific implementations
│   ├── layer_i8*.c            # Quantized layer implementations
│   ├── layers.h               # Layer interface definitions
│   ├── weigths_export.h       # Network weights for MSX
│   ├── zmeans.h               # Latent space cluster centers
│   ├── build.sh               # Build script (Linux)
│   └── build.bat              # Build script (Windows)
│
├── cluster/                    # Latent space analysis
│   ├── cluster.py             # K-means clustering on z-space
│   ├── compute_z_mean.py      # Compute characteristic vectors
│   ├── infer_classifier_batch.py  # Batch inference tool
│   └── data/                  # Cluster analysis data
│
├── infer/                      # Classifier inference tools
│   └── infer_classifier.py    # Portrait characteristic classifier
│
├── images/                     # Screenshots and demos
│
└── dgan.rom                    # Compiled MSX ROM cartridge
```

## Building the Project

### Prerequisites

- **For Training**:
  - Python 3.7+
  - PyTorch
  - torchvision
  - PIL/Pillow
  - NumPy

- **For MSX ROM**:
  - [MSXgl](https://github.com/aoineko-fr/MSXgl) development environment
  - SDCC compiler
  - Make

### Training the Network

```bash
cd train
python train_wgan_v4.py --dataset <path_to_dataset> --epochs 200
```

### Exporting Weights

After training, export the quantized weights for the C runtime:

```bash
cd runtime
python export_gl.py
```

### Building the ROM

```bash
cd src
./build.sh  # Linux/Mac
# or
build.bat   # Windows
```

This will generate `dgan.rom` that can be loaded into MSX hardware or emulators.

## Running on MSX

### Using an Emulator (openMSX)

```bash
openmsx -cart dgan.rom
```

### On Real Hardware

1. Flash the ROM to a compatible MSX cartridge
2. Insert the cartridge into your MSX computer
3. Power on the system

## Usage

1. **Selection Menu**: Use arrow keys to navigate through options
2. **Choose Characteristics**: Select desired portrait features
3. **Generate**: Press the action button to generate a portrait
4. **Random Mode**: Select "Random" to generate unpredictable portraits

## Latent Space Clustering

The `cluster/` directory contains tools for analyzing the latent space:

- Compute mean z-vectors for specific characteristics
- Perform k-means clustering to discover portrait variations
- Generate datasets for characteristic classification

These tools help in creating the selection menu by mapping user choices to specific regions of the latent space.

## Performance

- **Generation Time**: ~2-3 seconds per portrait on real MSX hardware
- **Memory Usage**: Optimized to fit within 64KB RAM constraints
- **ROM Size**: Approximately 32-48KB (including all weights)

## Screenshots

The `images/` directory contains screenshots showing:
- Generated portrait examples
- Selection menu interface
- Different characteristic combinations

## Technical Challenges Solved

1. **Extreme Quantization**: Reducing a neural network to 4-bit weights while maintaining quality
2. **Integer-Only Inference**: Implementing matrix operations without floating point
3. **Memory Constraints**: Fitting a complete GAN generator in <64KB
4. **Speed Optimization**: Making inference practical on a 3.58 MHz processor
5. **Latent Space Control**: Mapping discrete characteristics to continuous z-space

## Future Improvements

- Support for different screen modes (Screen 2, Screen 4)
- Color portrait generation
- Animation/morphing between portraits
- Save/load favorite portraits
- Expanded characteristic options

## License

This project is provided as-is for educational and personal use.

## Acknowledgments

- **MSXgl**: Excellent development framework for MSX
- **openMSX**: Essential for testing and development
- **PyTorch**: Training framework
- MSX community for hardware specifications and support

## Author

Created as an exploration of running modern neural networks on vintage 8-bit hardware.

---

*Bringing AI to the 1980s, one portrait at a time.*
