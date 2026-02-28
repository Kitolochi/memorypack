---
title: BERT and GPT - Two Paradigms
---

# BERT: Bidirectional Encoder Representations

BERT (Bidirectional Encoder Representations from Transformers) was introduced by Google in 2018. It uses only the encoder portion of the Transformer architecture and is trained with a masked language modeling objective. During pre-training, random tokens are masked and the model learns to predict them from the surrounding context.

BERT processes text bidirectionally, meaning it considers both left and right context simultaneously for every token. This is in contrast to autoregressive models that only look at preceding tokens. The bidirectional approach allows BERT to build richer contextual representations.

## Pre-training Objectives

BERT uses two pre-training tasks. The first is Masked Language Modeling (MLM), where 15% of tokens are selected for prediction. Of these, 80% are replaced with [MASK], 10% with random tokens, and 10% remain unchanged. The second task is Next Sentence Prediction (NSP), where the model predicts whether two sentences appear consecutively in the original text.

## BERT Variants

BERT-base has 12 layers, 768 hidden dimensions, and 12 attention heads, totaling 110 million parameters. BERT-large has 24 layers, 1024 hidden dimensions, and 16 attention heads, with 340 million parameters. Both were pre-trained on English Wikipedia and BooksCorpus.

Later variants include RoBERTa (which removes NSP and uses dynamic masking), ALBERT (which uses parameter sharing to reduce model size), and DeBERTa (which uses disentangled attention for position and content).

# GPT: Generative Pre-trained Transformer

GPT (Generative Pre-trained Transformer) was introduced by OpenAI and uses only the decoder portion of the Transformer. It is trained autoregressively — predicting the next token given all previous tokens. This makes GPT naturally suited for text generation tasks.

## Scaling Laws

The GPT series demonstrated that scaling model size, data, and compute leads to predictable performance improvements. GPT-2 had 1.5 billion parameters, GPT-3 had 175 billion, and GPT-4 is a mixture-of-experts model with an estimated 1.8 trillion parameters.

Research on scaling laws by Kaplan et al. showed that loss scales as a power law with model size, dataset size, and compute budget. The Chinchilla paper by Hoffmann et al. later showed that many models were over-parameterized relative to their training data, and that optimal compute allocation requires scaling data proportionally with model size.

## In-Context Learning

GPT-3 demonstrated that large language models can perform tasks by conditioning on a few examples provided in the prompt, without any gradient updates. This capability, called in-context learning or few-shot learning, emerges at sufficient model scale and allows a single model to handle diverse tasks.

Zero-shot performance — where the model performs a task with only a natural language instruction and no examples — also improves with scale. This suggests that large language models develop general-purpose reasoning abilities during pre-training.

## Instruction Tuning and RLHF

InstructGPT and ChatGPT refined GPT's capabilities using Reinforcement Learning from Human Feedback (RLHF). The process involves three stages: supervised fine-tuning on human-written demonstrations, training a reward model on human preference comparisons, and optimizing the policy using Proximal Policy Optimization (PPO).

This approach aligns the model's behavior with human intentions, making it more helpful, harmless, and honest. RLHF has become a standard technique for making language models suitable for real-world deployment.
