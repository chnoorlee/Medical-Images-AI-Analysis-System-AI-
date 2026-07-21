@echo off
REM ============================================================
REM  Medical AI — Complete Setup & Training Script for Windows
REM  RTX 5090 | CUDA 13.1 | Python 3.12
REM ============================================================
echo.
echo ============================================================
echo   MEDICAL AI — SETUP AND TRAINING
echo ============================================================
echo.

REM ── Step 1: Install CUDA PyTorch ─────────────────────────────
echo [1/4] Installing PyTorch with CUDA for RTX 5090...
echo        (This GPU needs PyTorch CUDA 13.0+ for Blackwell sm_120)
echo.
echo        Downloading ~2.6GB — this may take 10-30 minutes...
echo.

pip install --force-reinstall "torch>=2.13.0" "torchvision>=0.28.0" --index-url https://download.pytorch.org/whl/cu130

echo.
echo [CHECK] Verifying PyTorch GPU support...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only!')"

REM ── Step 2: Install Python dependencies ──────────────────────
echo.
echo [2/4] Installing training dependencies...
pip install medmnist scikit-learn tensorboard huggingface_hub tqdm --quiet

REM ── Step 3: Download datasets ────────────────────────────────
echo.
echo [3/4] Pre-downloading medical datasets...
python -c "import warnings; warnings.filterwarnings('ignore'); import medmnist; dss=['breastmnist','bloodmnist','octmnist','retinamnist','tissuemnist','dermamnist','pathmnist','chestmnist']; [print(f'  Downloading {d}...') or eval(medmnist.INFO[d]['python_class'])(split='train',download=True,size=28,root='./data') for d in dss]; print('All datasets ready!')"

REM ── Step 4: Train all models ─────────────────────────────────
echo.
echo [4/4] Starting full training pipeline...
echo ============================================================
echo.
python run_training.py
echo.
echo ============================================================
echo   DONE! Check trained_models/ for results
echo ============================================================
pause
