# SCRAP

Repo for paper [Self-Consistent Reasoning-based Aspect Sentiment Quad Prediction with Extract-Then-Assign Strategy](https://arxiv.org/abs/2403.00354)

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

### Citation

```sh
@inproceedings{kim-etal-2024-self-consistent,
    title = "Self-Consistent Reasoning-based Aspect-Sentiment Quad Prediction with Extract-Then-Assign Strategy",
    author = "Kim, Jieyong  and
      Heo, Ryang  and
      Seo, Yongsik  and
      Kang, SeongKu  and
      Yeo, Jinyoung  and
      Lee, Dongha",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2024",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.findings-acl.435/",
    doi = "10.18653/v1/2024.findings-acl.435",
    pages = "7295--7303",
    abstract = "In the task of aspect sentiment quad prediction (ASQP), generative methods for predicting sentiment quads have shown promisingresults. However, they still suffer from imprecise predictions and limited interpretability, caused by data scarcity and inadequate modeling of the quadruplet composition process. In this paper, we propose Self-Consistent Reasoning-based Aspect sentiment quadruple Prediction (SCRAP), optimizing its model to generate reasonings and the corresponding sentiment quadruplets in sequence. SCRAP adopts the Extract-Then-Assign reasoning strategy, which closely mimics human cognition. In the end, SCRAP significantly improves the model{'}s ability to handle complex reasoning tasks and correctly predict quadruplets through consistency voting, resulting in enhanced interpretability and accuracy in ASQP."
}
```
