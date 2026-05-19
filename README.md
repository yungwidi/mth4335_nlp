# ArXiv Abstract Domain Classification

This is my final project for MTH4335: Natural Language Processing.

Project goal: Classifying academic paper abstracts into their research domain using TF-IDF feature extraction and Logistic Regression.

## Overview

Given an arXiv paper's abstract, the model predicts which of four academic domains it belongs to: **Computer Science**, **Mathematics**, **Physics**, or **Statistics**.

The project uses the [`librarian-bots/arxiv-metadata-snapshot`](https://huggingface.co/datasets/librarian-bots/arxiv-metadata-snapshot) dataset from HuggingFace. After filtering to the four target domains and dropping unrelated fields (economics, electrical engineering, quantitative finance, etc.), around 2.7 million papers are used for training and evaluation.

## Approach

**Feature Extraction — TF-IDF (Term Frequency–Inverse Document Frequency)**

Each abstract is converted into a 20,000-dimensional sparse vector where each dimension represents a word or bigram from the vocabulary. TF-IDF scores words by how often they appear in a given abstract relative to how rare they are across the entire corpus. Domain-specific terms like "Hamiltonian" (Physics) or "theorem" (Math) receive high scores because they appear frequently within their domain but rarely elsewhere. Common words like "the" or "show" are effectively zeroed out.

**Classifier — Logistic Regression (SAGA solver)**

A multinomial Logistic Regression model learns a weight vector for each domain. Each weight captures how strongly a given word signals that domain. At prediction time, the model computes a dot product between the abstract's TF-IDF vector and each domain's weight vector, applies softmax to produce probabilities, and predicts the domain with the highest probability.

**Text Preprocessing**

Abstracts are cleaned by removing LaTeX commands and inline math, stripping non-ASCII characters, collapsing whitespace, and lowercasing. This normalization ensures consistent tokenization across the corpus.

**Domain Mapping**

ArXiv categories are automatically mapped to domains using the category prefix. Categories starting with `cs.` map to CS, `math.` and `math-ph` map to Math, `stat.` maps to Statistics, and a set of known physics prefixes (`hep-th`, `cond-mat`, `astro-ph`, `quant-ph`, `gr-qc`, etc.) map to Physics. Papers from other fields are excluded. Each paper is labeled by its primary (first-listed) category.

**Class distribution:**

| Domain | Description |
|---|---|
| Physics | High energy physics, condensed matter, astrophysics, quantum physics, nuclear physics, general physics |
| Math | Pure and applied mathematics, mathematical physics |
| CS | All computer science subfields (ML, CV, NLP, algorithms, systems, etc.) |
| Statistics | Statistical methodology, machine learning (stat), theory, applications |

## Requirements

```
python >= 3.10
datasets
scikit-learn
pandas
numpy
matplotlib
```

## Usage

Run the notebook end-to-end. The dataset is loaded directly from HuggingFace (requires internet on first run, cached afterward). No separate data download or preprocessing step is needed.
