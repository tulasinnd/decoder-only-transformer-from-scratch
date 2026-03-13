# decoder only transformer from scratch

## Introduction
This project implements a **Transformer Decoder architecture from scratch using PyTorch**. The objective of this repository is to gain a deep understanding of how modern language models work internally by manually building the core components of a decoder-only Transformer.

Instead of relying on high-level libraries, this implementation focuses on constructing each part of the architecture step by step, including token embeddings, positional encoding, masked multi-head self-attention, feed-forward networks, residual connections, and layer normalization.

Decoder-only Transformer architectures form the foundation of many modern large language models. By implementing these mechanisms from first principles, this project aims to build strong intuition about how tokens interact through attention, how autoregressive text generation works, and how such models are trained and used in practice.

## Project Goals
* Understand the full mechanism of Transformer models by implementing each core component from scratch, including the deeper internal layers.
* Build a complete end-to-end pipeline covering data processing, training, optimization, and inference.
* Develop intuition about how language is modeled within the architecture and identify the core mechanisms driving this process.
* Experiment with different configurations to observe failures, gain insights, and learn how model behavior can be improved.
* Continuously extend the project by adding new components, techniques, and architectural improvements over time.