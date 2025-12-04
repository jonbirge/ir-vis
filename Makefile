# Makefile for IR-to-Color Image Translation Project
# 
# Usage:
#   make setup     - Create virtual environment and install dependencies
#   make train     - Train the model with default settings
#   make clean     - Remove generated files (checkpoints, visualizations)
#   make help      - Show all available targets

# Configuration
PYTHON := python
VENV_DIR := venv
VENV_PYTHON := $(VENV_DIR)/Scripts/python
VENV_PIP := $(VENV_DIR)/Scripts/pip
OUTPUT_DIR := outputs
DATA_DIR := data

# Detect OS for activation script path
ifeq ($(OS),Windows_NT)
    VENV_ACTIVATE := $(VENV_DIR)/Scripts/activate
    RM_CMD := rmdir /s /q
    MKDIR_CMD := mkdir
else
    VENV_ACTIVATE := $(VENV_DIR)/bin/activate
    VENV_PYTHON := $(VENV_DIR)/bin/python
    VENV_PIP := $(VENV_DIR)/bin/pip
    RM_CMD := rm -rf
    MKDIR_CMD := mkdir -p
endif

.PHONY: help setup venv install train train-coco train-cityscapes \
        clean clean-checkpoints clean-visualizations clean-logs clean-data \
        test test-model test-dataset test-losses lint format \
        inference tensorboard

# Default target
help:
	@echo ============================================================
	@echo IR-to-Color Image Translation - Makefile Help
	@echo ============================================================
	@echo.
	@echo Setup targets:
	@echo   make setup              - Create venv and install all dependencies
	@echo   make install            - Install dependencies (assumes venv active)
	@echo   make venv               - Show how to activate the virtual environment
	@echo.
	@echo Training targets:
	@echo   make train              - Train with default settings (COCO dataset)
	@echo   make train-coco         - Train with COCO dataset (auto-downloads)
	@echo   make train-cityscapes   - Train with Cityscapes dataset
	@echo   make train-debug        - Quick training run for debugging (few samples)
	@echo   make resume             - Resume training from latest checkpoint
	@echo.
	@echo Inference targets:
	@echo   make inference          - Run inference on sample images
	@echo.
	@echo Testing targets:
	@echo   make test               - Run all tests
	@echo   make test-model         - Test model architecture
	@echo   make test-dataset       - Test dataset loading
	@echo   make test-losses        - Test loss functions
	@echo.
	@echo Cleaning targets:
	@echo   make clean              - Remove all generated files
	@echo   make clean-checkpoints  - Remove only checkpoints
	@echo   make clean-visualizations - Remove only visualizations
	@echo   make clean-logs         - Remove only logs
	@echo   make clean-data         - Remove downloaded datasets (CAREFUL!)
	@echo   make clean-all          - Remove everything including data
	@echo.
	@echo Utility targets:
	@echo   make lint               - Run code linting (if installed)
	@echo   make format             - Format code with black (if installed)
	@echo   make tensorboard        - Launch TensorBoard for log visualization
	@echo.

# ============================================================
# Setup Targets
# ============================================================

setup: $(VENV_DIR) install
	@echo.
	@echo ============================================================
	@echo Setup complete!
	@echo.
	@echo To activate the virtual environment:
ifeq ($(OS),Windows_NT)
	@echo   Windows CMD:        $(VENV_DIR)\Scripts\activate.bat
	@echo   Windows PowerShell: $(VENV_DIR)\Scripts\Activate.ps1
else
	@echo   source $(VENV_ACTIVATE)
endif
	@echo.
	@echo Then run: make train
	@echo ============================================================

$(VENV_DIR):
	@echo Creating virtual environment in $(VENV_DIR)...
	$(PYTHON) -m venv $(VENV_DIR)
	@echo Virtual environment created.

install:
	@echo Installing dependencies...
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install torch torchvision --index-url https://download.pytorch.org/whl/cu121
	$(VENV_PIP) install pillow numpy matplotlib tqdm requests
	@echo.
	@echo Dependencies installed successfully!

# Show activation instructions
venv:
	@echo ============================================================
	@echo To activate the virtual environment:
	@echo.
ifeq ($(OS),Windows_NT)
	@echo   Windows CMD:
	@echo     $(VENV_DIR)\Scripts\activate.bat
	@echo.
	@echo   Windows PowerShell:
	@echo     $(VENV_DIR)\Scripts\Activate.ps1
	@echo.
	@echo   Git Bash / MSYS2:
	@echo     source $(VENV_DIR)/Scripts/activate
else
	@echo   source $(VENV_ACTIVATE)
endif
	@echo.
	@echo To deactivate: deactivate
	@echo ============================================================

# ============================================================
# Training Targets
# ============================================================

train: train-coco

train-coco:
	@echo Starting training with COCO dataset...
	$(VENV_PYTHON) train.py --dataset coco

train-cityscapes:
	@echo Starting training with Cityscapes dataset...
	@echo NOTE: Cityscapes requires manual download. See dataset.py for instructions.
	$(VENV_PYTHON) train.py --dataset cityscapes

train-debug:
	@echo Starting debug training run (limited samples)...
	$(VENV_PYTHON) -c "from config import get_config; c = get_config(); c.data.max_train_samples = 100; c.training.num_epochs = 2; c.training.batch_size = 4; exec(open('train.py').read().replace('config = get_config()', 'config = c'))"
	@echo If the above doesn't work, modify config.py temporarily for debugging.

resume:
	@echo Resuming training from latest checkpoint...
	$(VENV_PYTHON) train.py

# Training with custom parameters
train-custom:
	@echo Usage: make train-custom EPOCHS=100 BATCH=16 LR=0.0001
	$(VENV_PYTHON) train.py --epochs $(EPOCHS) --batch-size $(BATCH) --lr $(LR)

# ============================================================
# Inference Targets
# ============================================================

inference:
	@echo Running inference...
	@echo Usage: make inference-single IR=path/to/ir.png REF=path/to/ref.png
	@echo        make inference-batch IR_DIR=path/to/ir_images REF_DIR=path/to/ref_images
	$(VENV_PYTHON) inference.py --checkpoint $(OUTPUT_DIR)/checkpoints/best_model.pt --help

inference-single:
	$(VENV_PYTHON) inference.py \
		--checkpoint $(OUTPUT_DIR)/checkpoints/best_model.pt \
		--ir $(IR) \
		--ref $(REF) \
		--output ./inference_results

inference-batch:
	$(VENV_PYTHON) inference.py \
		--checkpoint $(OUTPUT_DIR)/checkpoints/best_model.pt \
		--ir-dir $(IR_DIR) \
		--ref-dir $(REF_DIR) \
		--output ./inference_results

# ============================================================
# Testing Targets
# ============================================================

test:
	@echo Running all tests...
	$(VENV_PYTHON) tests.py

test-model:
	@echo Testing model architecture...
	$(VENV_PYTHON) tests.py TestModel

test-dataset:
	@echo Testing dataset loading...
	$(VENV_PYTHON) tests.py TestDataset

test-losses:
	@echo Testing loss functions...
	$(VENV_PYTHON) tests.py TestLosses

test-utils:
	@echo Testing utility functions...
	$(VENV_PYTHON) tests.py TestUtils

test-config:
	@echo Testing configuration...
	$(VENV_PYTHON) tests.py TestConfig

test-integration:
	@echo Running integration tests...
	$(VENV_PYTHON) tests.py TestIntegration

test-quick:
	@echo Running quick component tests (no integration)...
	$(VENV_PYTHON) tests.py TestConfig
	$(VENV_PYTHON) tests.py TestUtils

# ============================================================
# Cleaning Targets
# ============================================================

clean: clean-checkpoints clean-visualizations clean-logs
	@echo.
	@echo Cleaned generated files. Data preserved.
	@echo To also remove downloaded data: make clean-data

clean-checkpoints:
	@echo Removing checkpoints...
ifeq ($(OS),Windows_NT)
	@if exist "$(OUTPUT_DIR)\checkpoints" $(RM_CMD) "$(OUTPUT_DIR)\checkpoints" 2>nul || echo No checkpoints to remove
else
	$(RM_CMD) $(OUTPUT_DIR)/checkpoints 2>/dev/null || echo "No checkpoints to remove"
endif

clean-visualizations:
	@echo Removing visualizations...
ifeq ($(OS),Windows_NT)
	@if exist "$(OUTPUT_DIR)\visualizations" $(RM_CMD) "$(OUTPUT_DIR)\visualizations" 2>nul || echo No visualizations to remove
else
	$(RM_CMD) $(OUTPUT_DIR)/visualizations 2>/dev/null || echo "No visualizations to remove"
endif

clean-logs:
	@echo Removing logs...
ifeq ($(OS),Windows_NT)
	@if exist "$(OUTPUT_DIR)\logs" $(RM_CMD) "$(OUTPUT_DIR)\logs" 2>nul || echo No logs to remove
	@if exist "test_logs" $(RM_CMD) "test_logs" 2>nul || echo No test logs to remove
else
	$(RM_CMD) $(OUTPUT_DIR)/logs 2>/dev/null || echo "No logs to remove"
	$(RM_CMD) test_logs 2>/dev/null || echo "No test logs to remove"
endif

clean-inference:
	@echo Removing inference results...
ifeq ($(OS),Windows_NT)
	@if exist "inference_results" $(RM_CMD) "inference_results" 2>nul || echo No inference results to remove
else
	$(RM_CMD) inference_results 2>/dev/null || echo "No inference results to remove"
endif

clean-data:
	@echo ============================================================
	@echo WARNING: This will delete all downloaded datasets!
	@echo Press Ctrl+C to cancel, or wait 5 seconds to continue...
	@echo ============================================================
ifeq ($(OS),Windows_NT)
	@timeout /t 5 /nobreak >nul
	@if exist "$(DATA_DIR)" $(RM_CMD) "$(DATA_DIR)" 2>nul || echo No data to remove
else
	@sleep 5
	$(RM_CMD) $(DATA_DIR) 2>/dev/null || echo "No data to remove"
endif
	@echo Data removed.

clean-all: clean clean-data clean-inference
	@echo.
	@echo All generated files and data removed.

clean-venv:
	@echo Removing virtual environment...
ifeq ($(OS),Windows_NT)
	@if exist "$(VENV_DIR)" $(RM_CMD) "$(VENV_DIR)" 2>nul || echo No venv to remove
else
	$(RM_CMD) $(VENV_DIR) 2>/dev/null || echo "No venv to remove"
endif

# ============================================================
# Utility Targets
# ============================================================

lint:
	@echo Running linting...
	$(VENV_PIP) install flake8 --quiet 2>nul || true
	$(VENV_PYTHON) -m flake8 *.py --max-line-length=100 --ignore=E501,W503

format:
	@echo Formatting code with black...
	$(VENV_PIP) install black --quiet 2>nul || true
	$(VENV_PYTHON) -m black *.py --line-length=100

tensorboard:
	@echo Launching TensorBoard...
	@echo NOTE: TensorBoard integration requires adding TensorBoard logging to train.py
	$(VENV_PIP) install tensorboard --quiet 2>nul || true
	$(VENV_PYTHON) -m tensorboard.main --logdir=$(OUTPUT_DIR)/logs

# Show current configuration
show-config:
	$(VENV_PYTHON) config.py

# Count lines of code
loc:
	@echo Lines of code:
ifeq ($(OS),Windows_NT)
	@powershell -Command "Get-ChildItem *.py | ForEach-Object { $lines = (Get-Content $_.FullName | Measure-Object -Line).Lines; Write-Host ('{0,-20} {1,6} lines' -f $_.Name, $lines) }"
else
	@wc -l *.py | sort -n
endif

# Show GPU info
gpu-info:
	$(VENV_PYTHON) -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('Memory:', f'{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB' if torch.cuda.is_available() else 'N/A')"

# ============================================================
# Development Targets
# ============================================================

# Create requirements.txt from current environment
freeze:
	$(VENV_PIP) freeze > requirements.txt
	@echo Requirements saved to requirements.txt

# Install from requirements.txt
install-requirements:
	$(VENV_PIP) install -r requirements.txt
