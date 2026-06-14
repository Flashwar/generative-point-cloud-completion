#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Creating PPSurf venv..."
python3 -m venv --without-pip "$SCRIPT_DIR/pps_venv"
source "$SCRIPT_DIR/pps_venv/bin/activate"
curl -sS https://bootstrap.pypa.io/get-pip.py | python3

echo "Installing PyTorch..."
pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyG..."
# PyG wheels passend zur PyTorch/CUDA-Version
pip install torch-scatter torch-sparse torch-cluster torch-geometric \
    -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

echo "Installing remaining dependencies..."
pip install pytorch-lightning==2.* \
    numpy==1.* \
    scikit-learn \
    scikit-image==0.* \
    scipy==1.* \
    pandas==1.* \
    openpyxl \
    overrides==7.* \
    pykdtree==1.* \
    laspy==2.* \
    pillow \
    tqdm \
    pyglet==1.* \
    rtree==1.* \
    tensorboard \
    trimesh \
    pysdf \
    "jsonargparse[signatures]"

echo "✓ PPSurf venv ready"
echo "Activate with: source $SCRIPT_DIR/pps_venv/bin/activate"