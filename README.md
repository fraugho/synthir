# synthir

Generate synthetic information retrieval datasets with realistic user queries.

## Quick Start

```bash
# Install
cargo install --path .

# Generate a dataset (uses any OpenAI-compatible API)
synthir generate -t recipes -d 50 -q 100 \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --api-key none \
  -j 32
```

## Output Format

Simple flat structure (BEIR-compatible):

```
datasets/recipes/
├── corpus.jsonl    # Documents
├── queries.jsonl   # All queries (mixed types)
├── qrels.tsv       # Relevance judgments
└── metadata.json   # Generation info
```

## Commands

### `generate` - Create a new dataset

```bash
synthir generate [OPTIONS]

Options:
  -t, --topic <TOPIC>          Topic: recipes, academic, emails, legal, technical-docs, tiny-notes
  -d, --documents <N>          Number of documents [default: 100]
  -q, --queries-per-type <N>   Total queries (split evenly across types) [default: 500]
  --query-types <TYPES>        Comma-separated: natural,keyword,semantic,academic,complex,basic
  --base-url <URL>             LLM API URL [default: https://api.openai.com/v1]
  -m, --model <MODEL>          Model name [default: gpt-4]
  --api-key <KEY>              API key (or set OPENAI_API_KEY)
  -j, --concurrency <N>        Parallel requests [default: 1]
  --scoring-mode <MODE>        source, pooled, or exhaustive [default: source]
  -o, --output <DIR>           Output directory [default: ./datasets]
  --doc-diversity <MODE>       none, two-phase, or embedding [default: none]
  --query-diversity <MODE>     exact or embedding [default: exact]
  --diversity-threshold <N>    Similarity threshold 0.0-1.0 [default: 0.85]
  --doc-categories <LIST>      Comma-separated categories to cycle through
```

**Example:**
```bash
# Generate 100 docs with 30 queries (10 natural + 10 keyword + 10 semantic)
synthir generate -t tiny-notes -d 100 -q 30 \
  --query-types natural,keyword,semantic \
  --base-url http://192.168.0.133:8000/v1 \
  --model gpt-oss-120b \
  --api-key none \
  -j 128 \
  --scoring-mode exhaustive
```

### `remix` - Add new queries to existing dataset

Clone a dataset and generate new query types:

```bash
synthir remix --source ./NFCorpus --output-name nfcorpus-semantic \
  --query-types semantic \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --api-key none \
  -j 32
```

### `remix-batch` - Remix multiple datasets

```bash
synthir remix-batch --source ./EvalsDatasets \
  --query-types semantic \
  --output ../EvalsDatasets-semantic \
  --base-url http://192.168.0.133:8000/v1 \
  --model gpt-oss-120b \
  --api-key none \
  -j 128 \
  --scoring-mode exhaustive \
  --on-exist overwrite
```

## Query Types

| Type | Description | Example |
|------|-------------|---------|
| `natural` | Casual questions | "how do i fix a leaky faucet" |
| `keyword` | Search terms (may have typos) | "recipie chicken easy" |
| `semantic` | Zero word overlap | Doc: "Emperor Penguin" → "antarctic bird" |
| `basic` | Partial keyword match | Doc: "Emperor Penguin Encyclopedia" → "penguin book" |
| `academic` | Formal queries | "correlation between X and Y" |
| `complex` | Multi-hop reasoning | "difference between X and Y" |

## Scoring Modes

- **source**: 1-to-1 mapping (query → source document only)
- **pooled**: BM25 retrieves candidates, LLM scores them
- **exhaustive**: Score every query against every document (most thorough)

## Diversity Options

Control document and query diversity to avoid generating too-similar content.

### Document Diversity (`--doc-diversity`)

| Mode | Description | Speed |
|------|-------------|-------|
| `none` | No diversity checking (default, fastest) | Fast |
| `two-phase` | Generate idea/title first, check uniqueness, then expand | Medium |
| `embedding` | Reject documents with high embedding similarity | Slow |

### Query Diversity (`--query-diversity`)

| Mode | Description |
|------|-------------|
| `exact` | Exact string deduplication only (default) |
| `embedding` | Also reject semantically similar queries |

### Additional Options

```bash
--diversity-threshold <0.0-1.0>  # Similarity threshold (default: 0.85, higher = stricter)
--doc-categories <CATEGORIES>    # Force document variety (e.g., "soup,main,dessert,appetizer")
--embedding-url <URL>            # Separate embedding API (defaults to --base-url)
--embedding-model <MODEL>        # Embedding model (defaults to --model)
```

**Example with diversity:**
```bash
synthir generate -t recipes -d 100 -q 50 \
  --doc-diversity two-phase \
  --doc-categories "soup,salad,main,dessert,appetizer" \
  --query-diversity embedding \
  --diversity-threshold 0.8 \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  -j 32
```

## Embedding Support (for semantic queries)

When using `--scoring-mode pooled` with semantic queries, you can use a separate embedding model:

```bash
synthir remix-batch --source ./data \
  --query-types semantic \
  --base-url http://192.168.0.133:8000/v1 \
  --embedding-url http://localhost:8081/v1 \
  --model gpt-oss-120b \
  --embedding-model lfm \
  --scoring-mode pooled \
  --pool-size 30
```

## Local LLM Setup

Works with any OpenAI-compatible API:

**vLLM:**
```bash
vllm serve your-model --port 8000
```

**llama.cpp (for embeddings):**
```bash
llama-server -m model.gguf -c 32768 -np 32 --port 8081 --embeddings --pooling mean
```

## File Formats

**corpus.jsonl:**
```json
{"_id": "doc_001", "title": "Chocolate Cake", "text": "A rich chocolate cake..."}
```

**queries.jsonl:**
```json
{"_id": "natural_001", "text": "how to make chocolate cake moist"}
```

**qrels.tsv:**
```
query-id	doc-id	score
natural_001	doc_001	3
```

Scores: 0=not relevant, 1=marginal, 2=relevant, 3=highly relevant

## Features

- Query deduplication (regenerates if duplicate detected)
- Language detection (generates queries in detected language)
- Multi-locale support for international datasets
- Checkpoint/resume for long runs
- Concurrent generation with `-j` flag

## License

MIT
