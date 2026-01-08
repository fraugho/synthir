# synthir

A Rust CLI tool that generates high-quality synthetic information retrieval datasets with realistic user queries.

## Features

- **Realistic Query Generation**: Generates queries that mimic actual user search patterns, not polished academic queries
- **Multiple Query Types**: Natural, keyword, academic, complex, and mixed query styles
- **Concurrent Generation**: Parallel LLM requests with configurable concurrency (`-j` flag)
- **Benchmark Mode**: Find optimal concurrency settings for your LLM endpoint
- **Meta Mode**: Generate multiple datasets automatically with LLM-generated topics
- **Checkpointing**: Resume interrupted generation runs
- **Hard Negative Mining**: BM25 + LLM validation for challenging negative examples
- **BEIR Format**: Output compatible with standard IR evaluation frameworks

## Installation

```bash
cargo install --path .
```

Or build from source:

```bash
cargo build --release
./target/release/synthir --help
```

## Quick Start

```bash
# Generate a dataset with default settings
synthir generate -t recipes --api-key YOUR_API_KEY

# Generate with 8 concurrent requests
synthir generate -t recipes -j 8 --api-key YOUR_API_KEY

# Use with LMStudio or local LLM
synthir generate -t recipes \
  --base-url http://localhost:1234/v1 \
  --model local-model \
  --api-key lm-studio

# Benchmark to find optimal concurrency
synthir benchmark \
  --base-url http://localhost:1234/v1 \
  --model local-model \
  --api-key lm-studio

# Generate multiple datasets
synthir meta -t 5 -d 100 -q 500 -j 8 --api-key YOUR_API_KEY
```

## Commands

### `generate` - Generate a Single Dataset

```bash
synthir generate [OPTIONS]
```

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --topic` | Topic name (built-in or "llm-generated") | recipes |
| `-c, --corpus` | Path to existing corpus.jsonl | - |
| `-d, --documents` | Number of documents to generate | 100 |
| `-q, --queries-per-type` | Queries per query type | 500 |
| `-o, --output` | Output directory | ./datasets |
| `-m, --model` | Model identifier | gpt-4 |
| `--base-url` | LLM API base URL | https://api.openai.com/v1 |
| `--api-key` | API key (or set OPENAI_API_KEY) | - |
| `-j, --concurrency` | Concurrent LLM requests | 1 |
| `--resume` | Resume from checkpoint | false |
| `--dry-run` | Show what would happen | false |
| `--no-hard-negatives` | Skip hard negative mining | false |
| `--scoring-mode` | Scoring mode: `source` or `pooled` | source |
| `--pool-size` | Pool size for pooled scoring (top-k docs) | 30 |
| `-v, --verbose` | Verbose output | false |

### `queries` - Generate Queries for Existing Corpus

```bash
synthir queries -c ./corpus.jsonl -t natural -o ./output --api-key YOUR_KEY
```

| Option | Description | Default |
|--------|-------------|---------|
| `-c, --corpus` | Path to corpus.jsonl | required |
| `-t, --query-type` | Query type to generate | natural |
| `-n, --count` | Number of queries | 500 |
| `-o, --output` | Output directory | required |

### `benchmark` - Find Optimal Concurrency

```bash
synthir benchmark --base-url http://localhost:1234/v1 --api-key lm-studio
```

Output:
```
=== synthir Benchmark ===

Model: local-model
Endpoint: http://localhost:1234/v1

Latency Test Results (single request):
  Mean:  245ms
  P50:   230ms
  P95:   380ms
  P99:   520ms

Running throughput tests...

Concurrency |    Req/s |   Tokens/s | Avg Latency
------------------------------------------------
          1 |      4.1 |        820 |       245ms
          2 |      7.8 |       1560 |       256ms
          4 |     14.2 |       2840 |       282ms
          8 |     22.5 |       4500 |       355ms
         16 |     24.1 |       4820 |       663ms <-- diminishing returns
         32 |     23.8 |       4760 |      1344ms

Recommendation: Use --concurrency 8 for optimal throughput
```

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --samples` | Samples per concurrency level | 10 |
| `--levels` | Concurrency levels (comma-separated) | 1,2,4,8,16,32 |

### `meta` - Generate Multiple Datasets

```bash
synthir meta -t 5 -d 100 -q 500 --shared-corpus -j 8 --api-key YOUR_KEY
```

| Option | Description | Default |
|--------|-------------|---------|
| `-t, --topics` | Number of topics to generate | 5 |
| `-d, --documents` | Documents per topic | 100 |
| `-q, --queries-per-type` | Queries per type | 500 |
| `-o, --output` | Output directory | ./multi-datasets |
| `--shared-corpus` | Share corpus across topics | false |
| `-j, --concurrency` | Concurrent LLM requests | 1 |

### `topics` - List Available Topics

```bash
synthir topics
```

Built-in topics:
- `academic` - Academic papers and research
- `emails` - Professional and personal emails
- `legal` - Legal documents and contracts
- `miscellaneous` - Random documents
- `product-reviews` - E-commerce reviews
- `recipes` - Cooking recipes
- `technical-docs` - Technical documentation
- `tiny-notes` - Short personal notes

## Query Types

| Type | Description | Example |
|------|-------------|---------|
| `natural` | Casual Google-style questions | "how do i fix a leaky faucet" |
| `keyword` | Messy user searches (may have typos) | "recipie chicken easy" |
| `academic` | Formal, detailed queries | "What is the correlation between X and Y?" |
| `complex` | Multi-hop reasoning | "difference between sourdough and regular bread" |
| `mixed` | Random mix of all types | - |

## Output Format (BEIR Compatible)

```
datasets/
└── recipes/
    ├── corpus.jsonl          # Documents
    ├── queries/
    │   ├── natural/
    │   │   ├── queries.jsonl
    │   │   └── qrels.tsv
    │   ├── keyword/
    │   ├── academic/
    │   ├── complex/
    │   └── mixed/
    ├── merged/
    │   ├── general/          # All queries merged
    │   └── with-hard-negatives/
    └── metadata.json
```

### File Formats

**corpus.jsonl**:
```json
{"_id": "doc_000001", "title": "Chocolate Cake", "text": "A rich chocolate cake..."}
```

**queries.jsonl**:
```json
{"_id": "natural_000001", "text": "how to make chocolate cake moist"}
```

**qrels.tsv** (tab-separated):
```
query_id    doc_id      score
natural_000001  doc_000001  3
```

Relevance scores (TREC scale):
- 0 = Not relevant
- 1 = Marginally relevant
- 2 = Relevant
- 3 = Highly relevant

## Scoring Modes

synthir supports two scoring modes for generating relevance judgments:

### Source Mode (Default)

```bash
synthir generate -t recipes --scoring-mode source
```

- **1-to-1 mapping**: Each query maps to its source document only
- Faster generation
- Good for basic evaluation datasets

### Pooled Mode

```bash
synthir generate -t recipes --scoring-mode pooled --pool-size 30
```

- **Many-to-many mapping**: Each query can have multiple relevant documents
- Uses BM25 to retrieve top-k candidate documents per query
- LLM scores each candidate on 0-100 fine-grained scale
- **Cliff detection** finds natural score discontinuity:
  - Documents above cliff → TREC scores (1-3)
  - Documents below cliff → hard negatives (0)
- More realistic evaluation with graded relevance
- No artificial capping - cliff algorithm determines relevant set size

Example output distribution:
```
Query: "how to make chocolate cake moist"
  doc_000012: 92 → score 3 (highly relevant)
  doc_000045: 78 → score 3 (highly relevant)
  doc_000003: 65 → score 2 (relevant)
  --- cliff detected (gap: 25 points) ---
  doc_000089: 40 → score 0 (hard negative)
  doc_000023: 35 → score 0 (hard negative)
```

## Checkpointing

Generation progress is saved automatically. Resume with `--resume`:

```bash
# Start generation
synthir generate -t recipes -d 1000 -j 8 --api-key YOUR_KEY

# If interrupted, resume:
synthir generate -t recipes -d 1000 -j 8 --resume --api-key YOUR_KEY
```

## LLM Providers

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
synthir generate -t recipes
```

### LMStudio

```bash
synthir generate -t recipes \
  --base-url http://localhost:1234/v1 \
  --model local-model \
  --api-key lm-studio
```

### OpenRouter

```bash
synthir generate -t recipes \
  --base-url https://openrouter.ai/api/v1 \
  --model anthropic/claude-3-opus \
  --api-key YOUR_OPENROUTER_KEY
```

### Any OpenAI-Compatible API

```bash
synthir generate -t recipes \
  --base-url http://your-server/v1 \
  --model your-model \
  --api-key your-key
```

## Performance Tips

1. **Run benchmark first** to find optimal concurrency for your setup
2. **Use `-j 8` or higher** for local LLMs with good hardware
3. **Use `--shared-corpus`** in meta mode if topics are similar
4. **Use `--no-hard-negatives`** for faster generation (skip BM25 mining)

## License

MIT
