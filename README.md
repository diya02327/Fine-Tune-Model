# Fine-Tune-LLMs 🎯

This project focuses on fine-tuning a pre-trained transformer-based language model using the [IMDb movie reviews dataset](https://ai.stanford.edu/~amaas/data/sentiment/). The goal is to build a sentiment classification model that can accurately classify movie reviews as positive or negative.

## 🔍 Project Overview

- **Objective**: Fine-tune a DistilBERT model for binary sentiment classification
- **Dataset**: IMDb Large Movie Review Dataset (binary sentiment labels)
- **Model**: DistilBERT (or other Hugging Face transformer)
- **Framework**: PyTorch + Hugging Face Transformers
- **Tracking**: MLflow for experiment tracking
- **Evaluation**: Accuracy, F1-score, loss

---

## 📁 Directory Structure

```bash
Fine-Tune-Model/
│
├── data/                 # IMDb dataset or processed data
├── scripts/              # Python scripts for training, evaluation, etc.
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/               # Saved fine-tuned model files
├── notebooks/            # Jupyter notebooks for EDA or testing
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
