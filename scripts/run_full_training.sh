#!/bin/bash
# Full Medical AI Training Pipeline
# Trains all models for the medical image diagnosis system
# Usage: bash scripts/run_full_training.sh

set -e
cd "$(dirname "$0")/.."

echo "============================================"
echo "  MEDICAL AI - FULL TRAINING PIPELINE"
echo "  GPU: NVIDIA RTX 5090 (32GB VRAM)"
echo "  Date: $(date)"
echo "============================================"

# Phase 1: Small datasets for quick validation
echo ""
echo "[Phase 1] Training pneumonia detector (binary, fast)..."
python scripts/train_all.py --task pneumonia --epochs 30 --batch_size 64

echo ""
echo "[Phase 2] Training blood cell classifier (8 classes)..."
python scripts/train_all.py --task blood --epochs 40 --batch_size 48

# Phase 2: Medium complexity tasks
echo ""
echo "[Phase 3] Training skin lesion classifier (7 classes)..."
python scripts/train_all.py --task skin_lesion --epochs 50 --batch_size 48 --model efficientnet_b0

echo ""
echo "[Phase 4] Training retina OCT classifier (4 classes)..."
python scripts/train_all.py --task retina_oct --epochs 50 --batch_size 48 --model resnet50

# Phase 3: Complex tasks
echo ""
echo "[Phase 5] Training chest X-ray multi-label classifier (14 diseases)..."
python scripts/train_all.py --task chest_xray --epochs 60 --batch_size 32 --model densenet121 --input_size 320

echo ""
echo "[Phase 6] Training pathology classifier (9 classes)..."
python scripts/train_all.py --task pathology --epochs 50 --batch_size 32 --model densenet121

echo ""
echo "[Phase 7] Training tissue classifier (8 classes)..."
python scripts/train_all.py --task tissue --epochs 50 --batch_size 32 --model densenet121

echo ""
echo "============================================"
echo "  TRAINING COMPLETE!"
echo "  Models saved to: ./trained_models/"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Evaluate models: python -c '...'"
echo "  2. Upload to HuggingFace: python huggingface/upload_model.py --upload_all"
echo ""
