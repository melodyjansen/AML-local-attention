#  IMDb Sentiment Analysis with BERT: Full vs. Local Attention

This repository compares two variants of BERT-based models for sentiment analysis on the IMDb movie reviews dataset:

- `full_attention_baseline.ipynb`: Standard BERT with full self-attention  
- `local_attention_3_15_30.ipynb`: Custom BERT variant with **local self-attention**

The goal is to explore how restricting attention to a fixed local window affects model performance and efficiency.

---

##  Files

| File                          | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| `full_attention_baseline.ipynb` | Standard BERT model using Hugging Face Transformers for classification |
| `local_attention_3_15_30.ipynb`         | Custom BERT model with fixed-window local self-attention             |
| `README.md`                    | Current file                                                         |

---

##  Task

Binary **sentiment classification** on the [IMDb dataset](http://ai.stanford.edu/~amaas/data/sentiment/):

- **Positive (1)** or **Negative (0)** movie reviews  
- Text is tokenized using BERT tokenizer  

---

##  Models

###  Full Attention Baseline
- Based on Hugging Face's `BertForSequenceClassification`
- Standard global self-attention

###  Local Attention Model
- Modified `BertSelfAttention` with a **custom local attention mask**
- Each token attends only to a window of neighboring tokens (we experiment with window_size = 3, window_size = 15, and window_size = 30)

## Dataset

- **Dataset**: IMDB Movie Reviews
- **Task**: Binary sentiment classification (positive/negative)
- **Training samples**: 1,000 (subset for faster experimentation)
- **Test samples**: 200

## Training Configuration

- **Optimizer**: AdamW (lr=2e-5)
- **Epochs**: 3
- **Batch size**: 8
- **Device**: CUDA (GPU)
- **Loss function**: CrossEntropyLoss

## Results Summary

| Model | Accuracy | Notes |
|-------|----------|-------|
| **Full Attention Baseline** | **81%** | Standard BERT with full self-attention |
| Local Attention (window=3) | 49% | Very restrictive, poor performance |
| Local Attention (window=15) | 51% | Slight improvement but still limited |
| Local Attention (window=30) | **67%** | Best local attention performance |



