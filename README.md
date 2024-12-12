# French-Reduced-Word-Embedding

This repository contains the implementation of a project that compresses and improves word embeddings for the French language. Using FacebookAI's FastText embeddings as the baseline, we apply an AutoEncoder to reduce their size by 5 to 10 times and then fine-tune the compressed embeddings for enhanced performance.

## Overview

Word embeddings are critical for a variety of natural language processing tasks. However, high-dimensional embeddings can be computationally expensive and memory-intensive. This project addresses this challenge by:

1. Compressing FacebookAI's FastText embeddings for French using an AutoEncoder. Current FastText state-of-the-art is 300 dimensions. The goal is to make it 30 dimensions.
2. Fine-tuning the reduced embeddings on new data to improve their performance.

## Goals

1. **Compression**: Reduce the size of the FastText embeddings by a factor of 5 or 10 while preserving semantic quality.
2. **Fine-Tuning**: Enhance the compressed embeddings using additional training data.
3. **Evaluation**: Compare the compressed and fine-tuned embeddings against the original to ensure quality improvement.
