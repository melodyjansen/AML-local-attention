#  IMDb Sentiment Analysis with BERT: Full vs. Local Attention

This repository compares two variants of BERT-based models for sentiment analysis on the IMDb movie reviews dataset:

- `full_attention_baseline.ipynb`: Standard BERT with full self-attention  
- `local_attention.ipynb`: Custom BERT variant with **local self-attention**

The goal is to explore how restricting attention to a fixed local window affects model performance and efficiency.

---

##  Files

| File                          | Description                                                        |
|-------------------------------|--------------------------------------------------------------------|
| `full_attention_baseline.ipynb` | Standard BERT model using Hugging Face Transformers for classification |
| `local_attention.ipynb`         | Custom BERT model with fixed-window local self-attention             |
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
- Each token attends only to a window of neighboring tokens (e.g., ±3)



