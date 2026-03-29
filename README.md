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

| Methods            | HotpotQA CR | HotpotQA EM | HotpotQA F1 | 2Wiki CR | 2Wiki EM | 2Wiki F1 | MuSiQue CR | MuSiQue EM | MuSiQue F1 | NQ CR | NQ EM | NQ F1 | TriviaQA CR | TriviaQA EM | TriviaQA F1 |
|--------------------|-------------|-------------|-------------|----------|----------|----------|------------|------------|------------|-------|-------|-------|-------------|-------------|-------------|
| Direct             | –           | 19.32       | 26.87       | –        | 18.20    | 23.17    | –          | 2.77       | 8.99       | –     | 21.04 | 31.38 | –           | 55.01       | 61.87       |
| All Documents      | 1.00        | 26.97       | 38.17       | 1.00     | 14.00    | 24.83    | 1.00       | 13.94      | 24.57      | 1.00  | 33.99 | 47.55 | 1.00        | 60.98       | 70.83       |
| Top 5 Docs         | 5.80        | 26.30       | 37.18       | 6.01     | 11.60    | 22.42    | 4.00       | 6.74       | 13.36      | 5.80  | 32.36 | 45.54 | 5.80        | 57.69       | 67.50       |
| Top 10 Docs        | 2.90        | 27.21       | 38.54       | 3.00     | 13.40    | 23.97    | 2.00       | 10.50      | 19.43      | 2.90  | 33.24 | 47.59 | 2.90        | 59.85       | 69.46       |
| PolyCompressor(Ours)| 41.57       | 30.45       | 41.36       | 50.31    | 21.60    | 29.19    | 24.07      | 23.79      | 34.40      | 41.79 | 34.00 | 46.30 | 40.04       | 61.68       | 70.39       |

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
