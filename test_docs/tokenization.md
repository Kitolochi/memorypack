---
title: Tokenization in NLP
---

# Tokenization

Tokenization is the process of converting raw text into a sequence of tokens that can be processed by a language model. The choice of tokenization strategy significantly affects model performance, vocabulary size, and the ability to handle different languages.

## Byte-Pair Encoding (BPE)

BPE starts with individual characters and iteratively merges the most frequent pair of adjacent tokens. This process continues until a desired vocabulary size is reached. GPT-2 and GPT-3 use BPE with a vocabulary of approximately 50,000 tokens.

The merging process is trained on the model's training corpus, making the tokenization data-dependent. BPE handles rare and unseen words by decomposing them into known subword units. For example, "unhappiness" might be tokenized as "un" + "happiness" or "un" + "hap" + "piness" depending on the merge rules.

## WordPiece

WordPiece, used by BERT, is similar to BPE but uses a different criterion for selecting merges. Instead of frequency, WordPiece selects the pair that maximizes the likelihood of the training data. In practice, this often produces similar results to BPE but can better handle morphologically rich languages.

BERT's vocabulary contains 30,522 tokens. Unknown subwords are prefixed with "##" to indicate they are continuations (e.g., "playing" → "play" + "##ing"). This notation helps distinguish word-initial and word-internal subwords.

## SentencePiece and Unigram

SentencePiece treats the input as a raw byte stream rather than pre-tokenized words. This allows it to handle any language without requiring language-specific preprocessing. It supports both BPE and unigram tokenization algorithms.

The unigram model starts with a large vocabulary and iteratively removes tokens that minimize the impact on the training data likelihood. Each token has an associated probability, and tokenization finds the most probable segmentation using the Viterbi algorithm.

T5 and many multilingual models use SentencePiece with a vocabulary of 32,000 tokens. Llama models use SentencePiece with BPE and a vocabulary of 32,000 tokens.

## Tokenization Challenges

Different tokenization strategies handle numbers, code, and multilingual text differently. GPT tokenizers typically split numbers digit by digit, which makes arithmetic harder for the model. Some newer tokenizers include multi-digit number tokens.

Code tokenization is particularly challenging because programming languages have different syntax and naming conventions. Codex and Code Llama use tokenizers trained on code corpora to better handle programming constructs like indentation, operators, and variable names.

Multilingual tokenization must balance vocabulary space across languages. Models like mBERT and XLM-RoBERTa use shared vocabularies, but less-represented languages get fewer dedicated tokens, leading to longer tokenized sequences and degraded performance. Byte-level tokenizers like ByT5 avoid this issue entirely by operating on raw bytes, at the cost of longer sequences.
