# SCRAP

## Quick Start

First, transfer the data folder into the SCRAP folder.

Follow the steps below ⬇️

### Set Up

```sh
conda create -n scrap python=3.8
conda activate scrap
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```

### Train & Inference

run with command:

```sh
sh scripts/run_t5_base.sh
```

if you want to run T5-3B model:

```sh
sh scripts/run_t5_3b.sh
```
