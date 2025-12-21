#!/bin/bash
# Download RT-DETR checkpoint
# Usage: ./download_ckpt.sh

echo "Downloading checkpoint..."
mkdir -p weights

# Check if gdown is installed
if ! command -v gdown &> /dev/null
then
    echo "gdown could not be found. Installing..."
    pip install gdown
fi

gdown --id 10cuSaodCBDgVP-E5IEYTCB3F8za8VvUI -O weights/checkpoint0005.pth

echo "Done! Checkpoint saved to weights/checkpoint0005.pth"
