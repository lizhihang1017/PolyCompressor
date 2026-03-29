# Uncertainty-Aware Context Compression for Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2412.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2412.XXXXX)

A **training-free** framework that compresses retrieved contexts in Retrieval-Augmented Generation (RAG) by progressively reducing **epistemic**, **aleatoric**, and **logical uncertainties**. Achieves **97.48% token reduction** (~40× compression) while improving answer accuracy by **3.69 EM points** over uncompressed baselines.

> 🚀 **TL;DR**: PolyCompressor intelligently distills long, noisy retrieval results into concise, logically sufficient evidence without any training, dramatically reducing LLM token costs while preserving—and often improving—answer quality.

## 📌 Overview

Retrieval-Augmented Generation (RAG) systems often suffer from information overload: retrieving more documents increases recall but introduces noise, redundancy, and computational overhead. Existing compression methods act as black-box filters that ignore three critical uncertainties:

- **Epistemic Uncertainty**: Ambiguous user intent — what does the query *really* mean?
- **Aleatoric Uncertainty**: Noisy retrieval results — which documents are actually relevant?
- **Logical Uncertainty**: Insufficient evidence — does the compressed context support reasoning?

**PolyCompressor** addresses all three through a progressive, uncertainty-aware distillation pipeline:


## ✨ Key Features

- **🎯 Training-Free**: Works out-of-the-box with any LLM—no fine-tuning or gradient updates required
- **🔍 Uncertainty-Aware**: Explicitly models and reduces three types of uncertainty that plague RAG systems
- **📊 State-of-the-Art Performance**: 97.48% token reduction with 3.69 EM improvement over uncompressed baselines
- **🔌 Plug-and-Play**: Easy to integrate into existing RAG pipelines with minimal code changes
- **📈 Robust Scaling**: Maintains ~100 tokens regardless of input length, with improving accuracy as retrieval size grows

## 📊 Results

### Main Results

## Main Results

| Methods                       | HotpotQA |       |       | 2wikimultihop |       |       | MuSiQue |       |       | NQ    |       |       | TQA   |       |       |
|-------------------------------|----------|-------|-------|---------------|-------|-------|---------|-------|-------|-------|-------|-------|-------|-------|-------|
|                               | CR       | EM    | F1    | CR            | EM    | F1    | CR      | EM    | F1    | CR    | EM    | F1    | CR    | EM    | F1    |
| No Retrieval                  |          |       |       |               |       |       |         |       |       |       |       |       |       |       |       |
| Direct                        |          | 19.32 | 26.87 |               | 18.2  | 23.17 |         | 2.77  | 8.99  |       | 21.04 | 31.38 |       | 55.01 | 61.87 |
| Retrieval without Compression |          |       |       |               |       |       |         |       |       |       |       |       |       |       |       |
| All Documents                 | —        | 26.97 | 38.17 | —             | 14.00 | 24.83 | —       | 13.94 | 24.57 | —     | 33.99 | 47.55 | —     | 60.98 | 70.83 |
| Top 5 Documents               | 5.80     | 26.30 | 37.18 | 6.01          | 11.60 | 22.42 | 4.00    | 6.74  | 13.36 |       | 32.36 | 45.54 |       | 57.69 | 67.50 |
| Top 10 Documents              | 2.90     | 27.21 | 38.54 | 3.00          | 13.40 | 23.97 | 2.00    | 10.50 | 19.43 |       | 34.24 | 47.59 |       | 59.85 | 69.46 |
| Retrieval with Compression    |          |       |       |               |       |       |         |       |       |       |       |       |       |       |       |
| RECOMP (ICLR,24)              | 14.89    | 23.95 | 33.41 | 11.19         | 18.00 | 22.80 | 12.15   | 12.24 | 20.59 | 15.72 | 27.63 | 39.92 | 33.14 | 57.42 | 65.79 |
| LongLLMlingua (EMNLP,24)      | 9.39     | 26.73 | 38.13 | 9.71          | 16.20 | 23.55 | 8.25    | 8.56  | 18.27 | 9.33  | 26.66 | 39.63 | 9.43  | 59.72 | 68.06 |
| LongRefiner (ACL,25)          | 13.99    | 23.67 | 32.80 | 15.36         | 13.20 | 21.48 | 16.88   | 6.74  | 13.43 | 13.08 | 25.61 | 36.20 | 13.40 | 53.52 | 62.26 |
| CPC (AAAI,25)                 | 13.08    | 25.59 | 36.50 | 13.48         | 15.60 | 23.50 | 5.00    | 6.70  | 15.50 | 12.65 | 30.58 | 42.58 | 12.80 | 55.89 | 65.43 |
| Provence (ICLR,25)            | 11.69    | 26.13 | 37.05 | 16.95         | 16.60 | 23.46 | 18.25   | 14.43 | 24.81 | 8.21  | 33.33 | 46.76 | 8.70  | 59.21 | 68.81 |
| EXIT (ACL,25)                 | 8.11     | 28.85 | 40.73 | 11.29         | 16.00 | 27.94 | 10.33   | 21.88 | 33.12 | 3.80  | 33.36 | 46.05 | 6.00  | 59.49 | 69.75 |
| PolyCompressor                | 41.57    | 30.45 | 41.36 | 50.31         | 21.60 | 29.19 | 24.07   | 23.79 | 34.40 | 41.79 | 34.00 | 46.30 | 40.04 | 61.68 | 70.39 |


*Each cell reports Compression Rate (CR) / Exact Match (EM) / F1. CR is in × (original tokens / compressed tokens). “—” indicates no compression applied (CR not applicable).*

*PolyCompressor achieves superior compression while maintaining or improving answer quality across all benchmarks.*

### Document Reranking (NQ, Recall@1)

| Method | Recall@1 |
|--------|----------|
| BGE | 35.60% |
| SentenceBERT | 54.70% |
| BgeLLMEmb | 59.90% |
| BgeReranker | 62.90% |
| LongLLMLingua | 66.30% |
| **PolyCompressor (MVIG)** | **73.00%** |

### Ablation Study (NQ)

| MVIG | SP | DAKE | Tokens | EM | F1 |
|------|----|------|--------|-----|-----|
| ✗ | ✗ | ✗ | 701 | 15.25 | 26.13 |
| ✓ | ✗ | ✗ | 638 | 40.18 | 53.88 |
| ✓ | ✓ | ✗ | 352 | 41.00 | 52.72 |
| ✓ | ✓ | ✓ | **97** | **42.24** | **54.99** |

*Each component contributes to both compression and accuracy improvements.*

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/PolyCompressor.git
cd PolyCompressor
pip install -r requirements.txt
