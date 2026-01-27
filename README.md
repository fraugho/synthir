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
- **Multi-Locale Support**: Automatically processes all locales in multi-language datasets
- **Language-Aware Generation**: Detects language from existing queries (or documents) and generates new queries in that language

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
| `--trust-locale` | Trust locale directory names for language (e.g., fr-fr → French) | false |
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

**Multi-Locale Datasets:**

Datasets with locale subdirectories (e.g., `fr-fr`, `de-de`, `ja-jp`) are fully supported:

```
CCOCR/
├── ar-sa/
│   ├── label.json
│   └── queries.json
├── de-de/
├── es-es/
├── fr-fr/
├── it-it/
└── ja-jp/
```

All locales are automatically detected and processed. Query generation uses the appropriate language:

- **Default**: LLM detects language from existing queries (falls back to documents if no queries exist)
- **With `--trust-locale`**: Uses locale directory name (e.g., `fr-fr` → French)

```bash
# Auto-detect language via LLM (default)
synthir remix --source ./CCOCR --output-name CCOCR-semantic --api-key YOUR_KEY

# Trust locale names for faster processing
synthir remix --source ./CCOCR --output-name CCOCR-semantic --trust-locale --api-key YOUR_KEY
```

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
| `--trust-locale` | Trust locale directory names for language (e.g., fr-fr → French) | false |
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

### Query Generation Prompts

Below are the actual prompts used for each query type. Feedback welcome!

<details>
<summary><strong>natural</strong> - Casual Google-style questions</summary>

```
You generate REALISTIC Google search questions that regular people type.

Document:
"""
{document_text}
"""

Generate a casual question a normal person would Google:
- Simple, everyday language (NOT academic or formal)
- Like texting a friend or asking Google
- May be grammatically imperfect or lowercase
- Short and to the point (under 12 words usually)
- Examples of REAL natural searches:
  - "how do i fix a leaky faucet"
  - "is it bad to eat expired yogurt"
  - "why does my dog keep scratching"
  - "whats the best way to learn python"
  - "can you freeze cooked pasta"
  - "how long to boil eggs"

The question should be answerable by the document.

Output ONLY the question, nothing else.
```
</details>

<details>
<summary><strong>keyword</strong> - Messy user searches (may have typos)</summary>

```
You generate REALISTIC search queries that mimic how real users type.

Document:
"""
{document_text}
"""

Generate a keyword search like a REAL person would type into Google:
- 1-4 words, usually incomplete fragments
- May have typos or misspellings (about 15% of real searches do)
- No perfect grammar - just quick search terms
- Often missing words or using shorthand
- Examples of REAL keyword searches:
  - "pasta recipe easy"
  - "why wont car start"
  - "best laptop 2024"
  - "headache wont go away"
  - "recipie chicken" (typo)
  - "python tutorial beginer" (typo)
  - "how to"
  - "iphone not charging"

Output ONLY the search terms, nothing else. No quotes.
```
</details>

<details>
<summary><strong>academic</strong> - Formal, detailed queries</summary>

```
You generate detailed, academic-style queries for information retrieval.

Document:
"""
{document_text}
"""

Generate a formal, detailed query that a researcher or expert might use:
- Specific terminology and precise language
- Well-structured question or information need
- May reference specific concepts, methods, or metrics
- Longer and more detailed than typical user searches
- Examples of academic queries:
  - "What is the correlation between urban green space coverage and PM2.5 reduction in metropolitan areas?"
  - "How does the Maillard reaction temperature affect flavor compound formation in bread crusts?"
  - "What are the primary mechanisms by which SSRIs modulate serotonin reuptake in synaptic clefts?"
  - "How do transformer attention mechanisms compare to LSTM gates for sequence modeling?"

The document should contain information relevant to this query.

Output ONLY the query, nothing else.
```
</details>

<details>
<summary><strong>complex</strong> - Multi-hop reasoning queries</summary>

```
You generate search queries for information retrieval datasets.

Document:
"""
{document_text}
"""

Generate a complex query that requires reasoning to connect to this document.
Examples of complex queries:
- Comparative: "difference between X and Y"
- Multi-step: "how to do X after Y"
- Conditional: "best approach for X when Y"

The document should be relevant to answering this query, but not trivially so.

Output ONLY the query, nothing else. No quotes, no explanation.
```
</details>

<details>
<summary><strong>semantic</strong> - Zero lexical overlap (tests embeddings)</summary>

```
You generate document-finding queries that test SEMANTIC retrieval over lexical matching.

Document:
"""
{document_text}
"""

Generate a query someone would use to FIND this document, but with ZERO lexical overlap:
- This is a DOCUMENT SEARCH, not a question - you're looking for this document to exist
- Use BROADER CATEGORIES, SYNONYMS, or CONCEPTUAL descriptions
- A BM25/TF-IDF/keyword search MUST FAIL (no shared words or stems)
- Only an embedding-based search should find this document

CRITICAL RULES:
- Read the FULL document content above, not just any title
- NO words from the document (not even morphological variants like run/running)
- NO proper nouns, names, or specific terms from the document
- Think: "What category or concept does this document's CONTENT fall under?"
- Think: "What would someone search if they vaguely remembered what this was about?"

Examples (showing how content maps to queries):
- Doc about emperor penguins, their habitat, breeding -> Query: "bird books" or "antarctic wildlife"
- Doc explaining car maintenance steps for a sedan -> Query: "vehicle upkeep guide"
- Doc with French pastry recipes and techniques -> Query: "European dessert baking methods"
- Doc discussing Byzantine church construction -> Query: "eastern roman building styles"
- Doc teaching pandas, numpy, data analysis -> Query: "programming analytics manual"
- Doc about fixing air conditioning and heating -> Query: "residential climate control repair"

The query should be 2-6 words, like a library catalog search based on the document's content.

Output ONLY the query, nothing else.
```
</details>

<details>
<summary><strong>basic</strong> - Partial keyword matching</summary>

```
You generate document-finding queries with PARTIAL keyword matching.

Document:
"""
{document_text}
"""

Generate a query someone would use to FIND this document, with SOME but NOT ALL keywords matching:
- This is a DOCUMENT SEARCH - you're looking for this document
- Include 1-2 words that appear in the document
- OMIT other key identifying words to make it a partial match
- BM25/lexical search should find this, but not as the top result
- Query should be incomplete or abbreviated

RULES:
- Read the FULL document content above
- Pick the most descriptive 2-4 words, dropping some key terms
- Keep it like a quick, lazy search someone might type

Examples (showing partial keyword retention):
- Doc: "Encyclopedia of Emperor Penguins" -> Query: "penguin book" (dropped "emperor", "encyclopedia")
- Doc: "Toyota Camry 2019 Owner's Manual" -> Query: "camry manual" (dropped "toyota", "2019", "owner")
- Doc: "Introduction to Machine Learning with Python" -> Query: "machine learning python" (dropped "introduction")
- Doc: "The Complete Guide to Mediterranean Cooking" -> Query: "mediterranean recipes" (dropped "complete", "guide", changed "cooking")
- Doc: "Advanced Cardiovascular Life Support Manual" -> Query: "cardiac life support" (dropped "advanced", changed "cardiovascular")
- Doc about fixing iPhone battery issues -> Query: "iphone battery" (partial match)

The query should be 2-4 words with partial overlap.

Output ONLY the query, nothing else.
```
</details>

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

<details>
<summary><strong>Relevance Scoring Prompt</strong></summary>

```
You are a relevance judge for information retrieval.

Query: "{query}"

Document:
"""
{document_text}
"""

Rate the document's relevance to the query on a 0-3 scale:
0 = Not relevant (document does not help answer the query)
1 = Marginally relevant (mentions related concepts but doesn't answer)
2 = Relevant (partially answers or is useful for the query)
3 = Highly relevant (directly and completely answers the query)

Output ONLY a single digit (0, 1, 2, or 3). Nothing else.
```
</details>

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
