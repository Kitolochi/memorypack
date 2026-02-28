# memorypack

Compress a folder of markdown files into a small, structured summary that fits in an LLM's context window — without losing the ability to look up specifics.

## The Problem

You have a knowledge base — maybe 40 markdown files about your goals, health, career, projects, whatever. You want an AI to use all of it when answering questions. But dumping 80 raw text chunks into every prompt wastes tokens and most of it isn't relevant to the question.

## What This Does

memorypack reads your markdown files and produces a **three-tier summary**:

```
Tier 0 — Overview       (~200 tokens)   One paragraph covering everything.
Tier 1 — Topic summaries (~100 tokens each)   One paragraph per cluster of related content.
Tier 2 — Key facts      (bullet points)   Specific details extracted from each cluster.
```

The original files stay untouched. The summaries are an additional layer — you can still search the raw text when you need a specific detail.

## How It Works

There are 6 steps. Here's what each one does in plain English:

### Step 1: Chunk

Split every markdown file into small pieces (~512 tokens each). Break at headings and paragraph boundaries so each piece is about one thing.

### Step 2: Embed

Turn each chunk into a 384-number fingerprint (a vector) using a small local model (all-MiniLM-L6-v2). Chunks about similar topics get similar fingerprints.

### Step 3: Deduplicate

Compare every pair of chunks by their fingerprints. If two chunks are ≥92% similar, they're saying the same thing — keep the longer one, drop the other.

### Step 4: Cluster

Group the remaining chunks by similarity using k-means clustering. The algorithm automatically picks how many groups to make (2–10) by testing different values and keeping the one where groups are most internally coherent (silhouette score).

Each cluster gets a label pulled from the markdown headings of its chunks.

### Step 5: Summarize

Send each cluster to a summarizer to get a ~100 token paragraph. Then summarize all the summaries into one ~200 token overview.

### Step 6: Extract Facts

Scan each cluster for sentences that contain specific, useful information — things with proper nouns, numbers, or actionable language ("always", "must", "prefers"). Filter out vague transitions ("However, this...", "It should be noted..."). Keep up to 10 facts per cluster.

## Output

A single markdown file (or multi-file format) with this structure:

```markdown
# Knowledge Base: My Life
> Compressed by memorypack | 40 files | 8,820 → 1,400 tokens (6.3:1)

## Overview
One paragraph covering all major themes...

## Topics
### Health & Fitness
Summary paragraph about health-related content...

### Career & Projects
Summary paragraph about career-related content...

## Facts
### Health & Fitness
- Specific fact 1
- Specific fact 2

### Career & Projects
- Specific fact 1
- Specific fact 2
```

## Usage (Python CLI)

```bash
pip install -e .
memorypack ~/path/to/markdown/files -o compressed/
```

Options:
```
-o, --output DIR             Output directory (default: "compressed")
--format [single|multi]      One file or separate files
--device [cpu|cuda|mps]      GPU acceleration
--compression-target FLOAT   Target compression ratio (default: 6.0)
--chunk-size INT             Target tokens per chunk (default: 512)
--topic STR                  Name for the output header
```

## How It Fits Into a Larger System

memorypack produces static summaries. In a real app, you'd combine it with live vector search:

```
You ask: "How's my health progress?"

Three things happen:

1. COMPRESSED SUMMARIES (memorypack output)
   → Overview paragraph (always loaded, ~200 tokens)
   → Health cluster summary + facts (~150 tokens)

2. VECTOR SEARCH (RAG)
   → Embed your question
   → Find the 5-8 most similar raw chunks
   → But skip chunks that overlap with the summaries above

3. SCORED MEMORIES (separate system)
   → Short extracted facts from past conversations
   → Matched by keyword/topic relevance

All three get stuffed into the LLM prompt.
The summaries cover the broad picture cheaply.
The raw chunks fill in specific details the summaries missed.
Nothing is lost — the originals are always one search away.
```

### Adaptive RAG Budget

When summaries match the question well (high cosine similarity between question and cluster centroids), fewer raw chunks are needed:

| Summary match quality | Raw chunks fetched |
|----------------------|-------------------|
| Strong (sim ≥ 0.6)  | 5 chunks          |
| Medium (sim ≥ 0.45) | 8 chunks          |
| Weak (sim < 0.45)   | 12-15 chunks      |

This saves 500-1,500 tokens per query without losing coverage.

## Architecture

```
~/.claude/memory/
├── domains/
│   ├── health/
│   │   ├── profile.md
│   │   ├── goals.md
│   │   └── current_state.md
│   ├── career/
│   └── financial/
└── ...
        │
        ▼
┌──────────────┐
│   CHUNKER    │  Split by headings, ~512 tokens each
└──────┬───────┘
       ▼
┌──────────────┐
│  EMBEDDER    │  all-MiniLM-L6-v2, 384-dim vectors
└──────┬───────┘
       ▼
┌──────────────┐
│   DEDUP      │  Union-find, cosine ≥ 0.92 → keep longest
└──────┬───────┘
       ▼
┌──────────────┐
│  CLUSTER     │  K-means, auto-select k via silhouette
└──────┬───────┘
       ▼
┌──────────────┐
│ SUMMARIZE    │  ~100 tokens per cluster + ~200 token overview
└──────┬───────┘
       ▼
┌──────────────┐
│   FACTS      │  Rule-based extraction, max 10 per cluster
└──────┬───────┘
       ▼
  compressed-knowledge.json
  (or knowledge_base.md)
```

## Key Parameters

| Parameter | Default | What it does |
|-----------|---------|-------------|
| `chunk_size` | 512 | Target tokens per chunk |
| `dedup_threshold` | 0.92 | How similar two chunks must be to count as duplicates |
| `min_clusters` | 2 | Minimum number of topic groups |
| `max_clusters` | 10-20 | Maximum number of topic groups |
| `summary_max_tokens` | 150 | Length of each cluster summary |
| `overview_max_tokens` | 250 | Length of the overall overview |

## Dependencies

- **sentence-transformers** — local embeddings (no API calls)
- **transformers + torch** — BART summarization (local) or swap for an API-based summarizer
- **scikit-learn** — clustering algorithms
- **nltk** — sentence tokenization
- **click + rich** — CLI interface
