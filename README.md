# Uncertainty-Aware Context Compression for Retrieval-Augmented Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

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
<table border="1" cellpadding="5" cellspacing="0" style="border-collapse: collapse; text-align: center;">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Metric</th>
      <th>Direct</th>
      <th>All Documents</th>
      <th>Top 5 Docs</th>
      <th>Top 10 Docs</th>
      <th>PolyCompressor (Ours)</th>
    </tr>
  </thead>
  <tbody>
    <!-- HotpotQA -->
    <tr>
      <td rowspan="3">HotpotQA</td>
      <td>CR</td>
      <td>–</td>
      <td>1.00</td>
      <td><u>5.80</u></td>
      <td>2.90</td>
      <td><strong>41.57</strong></td>
    </tr>
    <tr>
      <td>EM</td>
      <td>19.32</td>
      <td>26.97</td>
      <td>26.30</td>
      <td><u>27.21</u></td>
      <td><strong>30.45</strong></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>26.87</td>
      <td>38.17</td>
      <td>37.18</td>
      <td><u>38.54</u></td>
      <td><strong>41.36</strong></td>
    </tr>
    <!-- 2Wiki -->
    <tr>
      <td rowspan="3">2Wiki</td>
      <td>CR</td>
      <td>–</td>
      <td>1.00</td>
      <td><u>6.01</u></td>
      <td>3.00</td>
      <td><strong>50.31</strong></td>
    </tr>
    <tr>
      <td>EM</td>
      <td><u>18.20</u></td>
      <td>14.00</td>
      <td>11.60</td>
      <td>13.40</td>
      <td><strong>21.60</strong></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>23.17</td>
      <td><u>24.83</u></td>
      <td>22.42</td>
      <td>23.97</td>
      <td><strong>29.19</strong></td>
    </tr>
    <!-- MuSiQue -->
    <tr>
      <td rowspan="3">MuSiQue</td>
      <td>CR</td>
      <td>–</td>
      <td>1.00</td>
      <td><u>4.00</u></td>
      <td>2.00</td>
      <td><strong>24.07</strong></td>
    </tr>
    <tr>
      <td>EM</td>
      <td>2.77</td>
      <td><u>13.94</u></td>
      <td>6.74</td>
      <td>10.50</td>
      <td><strong>23.79</strong></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>8.99</td>
      <td><u>24.57</u></td>
      <td>13.36</td>
      <td>19.43</td>
      <td><strong>34.40</strong></td>
    </tr>
    <!-- NQ -->
    <tr>
      <td rowspan="3">NQ</td>
      <td>CR</td>
      <td>–</td>
      <td>1.00</td>
      <td><u>5.80</u></td>
      <td>2.90</td>
      <td><strong>41.79</strong></td>
    </tr>
    <tr>
      <td>EM</td>
      <td>21.04</td>
      <td><u>33.99</u></td>
      <td>32.36</td>
      <td>33.24</td>
      <td><strong>34.00</strong></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>31.38</td>
      <td><u>47.55</u></td>
      <td>45.54</td>
      <td><strong>47.59</strong></td>
      <td>46.30</td>
    </tr>
    <!-- TriviaQA -->
    <tr>
      <td rowspan="3">TriviaQA</td>
      <td>CR</td>
      <td>–</td>
      <td>1.00</td>
      <td><u>5.80</u></td>
      <td>2.90</td>
      <td><strong>40.04</strong></td>
    </tr>
    <tr>
      <td>EM</td>
      <td>55.01</td>
      <td><u>60.98</u></td>
      <td>57.69</td>
      <td>59.85</td>
      <td><strong>61.68</strong></td>
    </tr>
    <tr>
      <td>F1</td>
      <td>61.87</td>
      <td><strong>70.83</strong></td>
      <td>67.50</td>
      <td>69.46</td>
      <td><u>70.39</u></td>
    </tr>
  </tbody>
</table>

*PolyCompressor achieves superior compression while maintaining or improving answer quality across all benchmarks.*

### Document Reranking (NQ，Recall@K)

| Methods               | @1   | @2   | @3   | @4   | @5   | @10  | @15  | @20  |
|-----------------------|------|------|------|------|------|------|------|------|
| LLMLingua             | 4.40 | 7.10 | 9.30 | 12.20 | 14.80 | 29.30 | 47.40 | 100.00 |
| BM25                  | 8.00 | 13.90 | 19.60 | 24.60 | 29.40 | 52.10 | 73.50 | 100.00 |
| BGE                   | 35.60 | 49.50 | 57.80 | 63.80 | 69.00 | 84.40 | 92.40 | 100.00 |
| Gzip                  | 50.10 | 55.70 | 59.70 | 63.00 | 65.80 | 77.20 | 87.60 | 100.00 |
| Jina                  | 52.20 | 66.00 | 74.00 | 78.60 | 82.20 | 92.20 | 97.00 | 100.00 |
| SentenceBert          | 54.70 | 66.60 | 73.00 | 77.50 | 81.30 | 91.50 | 96.20 | 100.00 |
| BgeLLMEmb             | 59.90 | 73.30 | 80.30 | 85.10 | 87.80 | 94.60 | 98.10 | 100.00 |
| BgeReranker           | 62.90 | 74.50 | 80.00 | 83.40 | 85.80 | 93.90 | 97.90 | 100.00 |
| LongLLMLingua         | 66.30 | 77.50 | 82.70 | 86.10 | 88.40 | 95.20 | 98.50 | 100.00 |
| Perception            | 70.50 | 80.50 | 85.00 | 87.80 | 89.00 | 94.70 | 98.30 | 100.00 |
| **PolyCompressor (MVIG)**       | **73.00** | **81.90** | **86.30** | **88.70** | **90.30** | **95.40** | **98.50** | **100.00** |

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/PolyCompressor.git
cd PolyCompressor
pip install -r requirements.txt
