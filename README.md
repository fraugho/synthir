# synthir

A Rust CLI tool that generates high-quality synthetic information retrieval datasets with realistic user queries.

## Features

- **Realistic Query Generation**: Generates queries that mimic actual user search patterns, not polished academic queries
- **Multiple Query Types**: Natural, keyword, academic, complex, semantic, basic, and mixed query styles
- **Dataset Remixing**: Clone existing datasets and replace queries/qrels with new types
- **OCR Dataset Support**: Auto-detects and handles both BEIR and OCR (GoodNotes-style) formats
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

# Remix an existing dataset with semantic queries
synthir remix --source ./NFCorpus --output-name semantic_nfcorpus --api-key YOUR_KEY

# Replace queries in-place (for fixing malformed datasets)
synthir remix --source ./GoodNotesOCR --in-place --query-types basic --api-key YOUR_KEY
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
| `--query-types` | Query types (comma-separated) | all |
| `-o, --output` | Output directory | ./datasets |
| `-m, --model` | Model identifier | gpt-4 |
| `--base-url` | LLM API base URL | https://api.openai.com/v1 |
| `--api-key` | API key (or set OPENAI_API_KEY) | - |
| `-j, --concurrency` | Concurrent LLM requests | 1 |
| `--resume` | Resume from checkpoint | false |
| `--dry-run` | Show what would happen | false |
| `--no-hard-negatives` | Skip hard negative mining | false |
| `--scoring-mode` | Scoring mode: `source`, `pooled`, or `exhaustive` | source |
| `--pool-size` | Pool size for pooled scoring (top-k docs) | 30 |
| `--score-scale` | Score scale: `trec` (0-3) or `range` (custom) | trec |
| `--score-min` | Minimum score for custom range | 0 |
| `--score-max` | Maximum score for custom range | 100 |
| `-v, --verbose` | Verbose output | false |

### `remix` - Remix Existing Dataset with New Queries

Clone an existing dataset and replace its queries/qrels, or modify in-place.

```bash
synthir remix [OPTIONS] --source <SOURCE>
```

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --source` | Path to source dataset (auto-detects format) | required |
| `-n, --output-name` | Name for output dataset | - |
| `--in-place` | Modify source dataset directly | false |
| `-o, --output` | Output directory for cloned dataset | source's parent |
| `--query-types` | Query types to generate | semantic |
| `-q, --queries-per-type` | Number of queries (or auto from source) | - |
| `-j, --concurrency` | Concurrent LLM requests | 1 |
| `--scoring-mode` | Scoring mode: `source`, `pooled`, or `exhaustive` | source |
| `--pool-size` | Pool size for pooled scoring | 30 |
| `--dry-run` | Preview what would happen | false |

**Examples:**

```bash
# Create semantic variant of NFCorpus (output: ./semantic_nfcorpus next to source)
synthir remix --source ./NFCorpus --output-name semantic_nfcorpus --api-key YOUR_KEY

# Create basic (partial keyword) variant with pooled scoring
synthir remix --source ./NFCorpus --output-name basic_nfcorpus \
  --query-types basic --scoring-mode pooled --api-key YOUR_KEY

# Replace queries in-place (useful for fixing malformed datasets)
synthir remix --source ./GoodNotesOCR --in-place --query-types semantic --api-key YOUR_KEY

# Output to specific directory instead of next to source
synthir remix --source ./NFCorpus --output-name test --output ./my-datasets --api-key YOUR_KEY

# Preview changes without executing
synthir remix --source ./NFCorpus --output-name test --dry-run
```

**Supported Formats:**

- **BEIR**: `corpus.jsonl`, `queries.jsonl`, `qrels/*.tsv`
- **OCR**: `label.json`, `queries.json` (GoodNotes-style)

The tool auto-detects the format and preserves qrels splits (train/dev/test) with original ratios.

**Cross-Platform Compatibility:**

All file readers handle different line endings automatically:
- Unix/Linux: LF (`\n`)
- Windows: CRLF (`\r\n`)
- Classic Mac: CR (`\r`)

This ensures datasets created on any platform can be read correctly.

### `remix-batch` - Batch Remix Multiple Datasets

Remix all compatible datasets in a directory with the same configuration.

```bash
synthir remix-batch [OPTIONS] --source <DIRECTORY>
```

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --source` | Directory containing multiple datasets | required |
| `--output-mode` | `sibling` (next to originals) or `grouped` (separate dir) | sibling |
| `-o, --output` | Output directory for grouped mode | `<source>-<query-type>` |
| `--on-exist` | Behavior for existing output: `skip`, `overwrite`, or `ask` | skip |
| `--query-types` | Query types to generate | semantic |
| `-q, --queries-per-type` | Number of queries (or auto from source) | - |
| `-j, --concurrency` | Concurrent LLM requests | 1 |
| `--scoring-mode` | Scoring mode: `source`, `pooled`, or `exhaustive` | source |
| `--pool-size` | Pool size for pooled scoring | 30 |
| `--dry-run` | Preview what would happen | false |

**Output Naming:** `{original-name}-{query-type}` (e.g., `WindowsAppSearch-semantic`)

**Examples:**

```bash
# Remix all datasets in EvalsDatasets with semantic queries (output next to originals)
synthir remix-batch --source ./EvalsDatasets --query-types semantic \
  --base-url http://localhost:8080/v1 --model local -j 32 --api-key none

# Put remixed datasets in a separate directory (default: EvalsDatasets-semantic)
synthir remix-batch --source ./EvalsDatasets --output-mode grouped \
  --query-types semantic --api-key YOUR_KEY

# Specify custom output directory
synthir remix-batch --source ./EvalsDatasets --output-mode grouped \
  --output ./my-remixed-datasets --query-types semantic --api-key YOUR_KEY

# Skip datasets that already have output
synthir remix-batch --source ./EvalsDatasets --on-exist skip --api-key YOUR_KEY

# Overwrite existing outputs
synthir remix-batch --source ./EvalsDatasets --on-exist overwrite --api-key YOUR_KEY

# Preview what would be processed
synthir remix-batch --source ./EvalsDatasets --dry-run
```

**Output Modes:**

*Sibling Mode (default):* New datasets placed next to originals
```
EvalsDatasets/
├── WindowsAppSearch/           # original
├── WindowsAppSearch-semantic/  # new
├── GoodNotesOCR/               # original
└── GoodNotesOCR-semantic/      # new
```

*Grouped Mode:* New datasets in separate directory
```
EvalsDatasets/           # originals untouched
EvalsDatasets-semantic/  # auto-named from query type (or custom --output)
├── WindowsAppSearch-semantic/
└── GoodNotesOCR-semantic/
```

The command automatically skips incompatible directories (no corpus/label files) and continues processing remaining datasets.

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
| `semantic` | Zero lexical overlap (tests embeddings) | Doc: "Emperor Penguin Encyclopedia" → "bird books" |
| `basic` | Partial keyword matching | Doc: "Emperor Penguin Encyclopedia" → "penguin book" |
| `mixed` | Random mix of all types | - |

### Semantic vs Basic Query Types

Both `semantic` and `basic` are designed to test retrieval beyond exact keyword matching:

**Semantic** (zero lexical overlap):
- BM25/TF-IDF should **fail** to find these
- Only embedding-based retrieval should succeed
- Example: "Encyclopedia of Emperor Penguins" → "bird books" or "antarctic wildlife reference"

**Basic** (partial keyword overlap):
- BM25/TF-IDF should find these, but not as top result
- Simulates lazy/abbreviated searches
- Example: "Encyclopedia of Emperor Penguins" → "penguin book" (dropped "emperor", "encyclopedia")

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
    │   ├── semantic/
    │   ├── basic/
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

Relevance scores (TREC scale, default):
- 0 = Not relevant
- 1 = Marginally relevant
- 2 = Relevant
- 3 = Highly relevant

## Score Scales

synthir supports two score scales for relevance judgments:

### TREC Scale (Default)

```bash
synthir generate -t recipes --score-scale trec
```

Standard TREC 0-3 relevance scale used in academic IR evaluation.

### Custom Range Scale

```bash
# 0-10 scale
synthir generate -t recipes --score-scale range --score-min 0 --score-max 10

# 1-5 star rating scale
synthir generate -t recipes --score-scale range --score-min 1 --score-max 5

# 0-100 percentage scale
synthir generate -t recipes --score-scale range --score-min 0 --score-max 100
```

Custom range scoring asks the LLM to rate relevance on your specified scale. The prompt dynamically adjusts thresholds based on your range. Useful for:
- Matching existing annotation schemes
- Fine-grained relevance distinctions
- Application-specific scoring needs

## Scoring Modes

synthir supports three scoring modes for generating relevance judgments:

### Source Mode (Default)

```bash
synthir generate -t recipes --scoring-mode source
```

- **1-to-1 mapping**: Each query maps to its source document only
- Fastest generation
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

### Exhaustive Mode

```bash
synthir generate -t recipes --scoring-mode exhaustive
```

- **Complete coverage**: Every query scored against every document
- O(queries × documents) LLM calls - use for smaller datasets
- Most thorough relevance judgments
- Best for high-quality evaluation benchmarks

Example output (5 docs, 12 queries = 60 qrels):
```
natural_000001 → doc_000001: 2, doc_000002: 3, doc_000003: 2, doc_000004: 1, doc_000005: 3
natural_000002 → doc_000001: 0, doc_000002: 3, doc_000003: 1, doc_000004: 2, doc_000005: 0
...
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

## Multi-Endpoint Support

For true parallel throughput with LMStudio, run multiple LMStudio server processes:

```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 lms server start --port 1234

# Terminal 2 (GPU 1)
CUDA_VISIBLE_DEVICES=1 lms server start --port 1235
```

Then use comma-separated URLs:

```bash
synthir generate -t recipes -j 8 \
  --base-url "http://localhost:1234/v1,http://localhost:1235/v1" \
  --model your-model \
  --api-key lm-studio
```

Requests are round-robined across all endpoints for maximum throughput.

**Note:** A single LMStudio server processes requests sequentially on the GPU, so increasing `-j` beyond 1-2 won't improve throughput. For true parallelism, use multiple server processes on different GPUs or ports.

For backends that support concurrent batch inference (vLLM, TGI, llama.cpp server with multiple slots), a single endpoint with high `-j` values will work.

## Output Structure

```
datasets/
└── recipes/
    ├── corpus.jsonl
    ├── queries/
    │   ├── natural/
    │   │   ├── queries.jsonl
    │   │   └── qrels.tsv
    │   ├── keyword/
    │   ├── academic/
    │   ├── complex/
    │   ├── semantic/
    │   ├── basic/
    │   └── mixed/
    ├── merged/
    │   ├── general/
    │   └── with-hard-negatives/
    ├── combined/              # Everything in one place
    │   ├── corpus.jsonl
    │   ├── queries.jsonl
    │   └── qrels.tsv
    └── metadata.json
```

The `combined/` folder contains all data in one place for easy use.

## Performance Tips

1. **Run benchmark first** to find optimal concurrency for your setup
2. **Use `-j 8` or higher** for local LLMs with good hardware
3. **Use multiple endpoints** for true parallel throughput with LMStudio
4. **Use `--shared-corpus`** in meta mode if topics are similar
5. **Use `--no-hard-negatives`** for faster generation (skip BM25 mining)
6. **Use smaller models** like `liquid/lfm2.5-1.2b` for faster iteration

## License

MIT
