use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Document diversity mode - how to ensure unique documents
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DocDiversity {
    /// No diversity checking (fastest, may produce similar docs)
    #[default]
    None,
    /// Two-phase: generate title first, check for duplicates, then expand
    TwoPhase,
    /// Embedding-based: reject docs too similar to existing ones
    Embedding,
}

impl std::fmt::Display for DocDiversity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocDiversity::None => write!(f, "none"),
            DocDiversity::TwoPhase => write!(f, "two-phase"),
            DocDiversity::Embedding => write!(f, "embedding"),
        }
    }
}

impl std::str::FromStr for DocDiversity {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().replace('_', "-").as_str() {
            "none" => Ok(DocDiversity::None),
            "two-phase" | "twophase" => Ok(DocDiversity::TwoPhase),
            "embedding" => Ok(DocDiversity::Embedding),
            _ => Err(format!(
                "Invalid doc diversity mode: {}. Use 'none', 'two-phase', or 'embedding'",
                s
            )),
        }
    }
}

/// Query diversity mode - how to ensure unique queries
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryDiversity {
    /// Exact string deduplication only (current behavior)
    #[default]
    Exact,
    /// Embedding-based: reject queries too similar to existing ones
    Embedding,
}

impl std::fmt::Display for QueryDiversity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryDiversity::Exact => write!(f, "exact"),
            QueryDiversity::Embedding => write!(f, "embedding"),
        }
    }
}

impl std::str::FromStr for QueryDiversity {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "exact" => Ok(QueryDiversity::Exact),
            "embedding" => Ok(QueryDiversity::Embedding),
            _ => Err(format!(
                "Invalid query diversity mode: {}. Use 'exact' or 'embedding'",
                s
            )),
        }
    }
}

/// Scoring mode for relevance judgments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoringMode {
    /// Only score query against its source document (1-to-1)
    #[default]
    Source,
    /// Use BM25 pooling + cliff detection for many-to-many mappings
    Pooled,
    /// Score every query against every document (exhaustive, expensive)
    Exhaustive,
}

/// Score scale for relevance judgments
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ScoreScale {
    /// TREC-style 0-3 scale (default)
    #[default]
    Trec,
    /// Custom range specified by min/max
    Range,
}

impl std::fmt::Display for ScoreScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoreScale::Trec => write!(f, "trec"),
            ScoreScale::Range => write!(f, "range"),
        }
    }
}

impl std::str::FromStr for ScoreScale {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "trec" => Ok(ScoreScale::Trec),
            "range" => Ok(ScoreScale::Range),
            _ => Err(format!(
                "Invalid score scale: {}. Use 'trec' or 'range'",
                s
            )),
        }
    }
}

impl std::fmt::Display for ScoringMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoringMode::Source => write!(f, "source"),
            ScoringMode::Pooled => write!(f, "pooled"),
            ScoringMode::Exhaustive => write!(f, "exhaustive"),
        }
    }
}

impl std::str::FromStr for ScoringMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "source" => Ok(ScoringMode::Source),
            "pooled" => Ok(ScoringMode::Pooled),
            "exhaustive" => Ok(ScoringMode::Exhaustive),
            _ => Err(format!("Invalid scoring mode: {}. Use 'source', 'pooled', or 'exhaustive'", s)),
        }
    }
}

/// Output mode for batch remix
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OutputMode {
    /// Create output datasets as siblings to originals (in same directory)
    #[default]
    Sibling,
    /// Create output datasets in a separate grouped directory
    Grouped,
}

impl std::fmt::Display for OutputMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OutputMode::Sibling => write!(f, "sibling"),
            OutputMode::Grouped => write!(f, "grouped"),
        }
    }
}

impl std::str::FromStr for OutputMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "sibling" => Ok(OutputMode::Sibling),
            "grouped" => Ok(OutputMode::Grouped),
            _ => Err(format!("Invalid output mode: {}. Use 'sibling' or 'grouped'", s)),
        }
    }
}

/// Behavior when output already exists
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OnExist {
    /// Skip datasets that already have output
    #[default]
    Skip,
    /// Overwrite existing output
    Overwrite,
    /// Ask user for each existing output
    Ask,
}

impl std::fmt::Display for OnExist {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OnExist::Skip => write!(f, "skip"),
            OnExist::Overwrite => write!(f, "overwrite"),
            OnExist::Ask => write!(f, "ask"),
        }
    }
}

impl std::str::FromStr for OnExist {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "skip" => Ok(OnExist::Skip),
            "overwrite" => Ok(OnExist::Overwrite),
            "ask" => Ok(OnExist::Ask),
            _ => Err(format!("Invalid on-exist mode: {}. Use 'skip', 'overwrite', or 'ask'", s)),
        }
    }
}

/// Query types supported by the generator
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum QueryType {
    /// Casual Google-style questions (e.g., "why is my pasta sticky")
    Natural,
    /// Real messy user keyword searches (e.g., "recipie pasta", "how to")
    Keyword,
    /// Formal, academic-style queries (e.g., "How does X affect Y in Z conditions?")
    Academic,
    /// Multi-hop reasoning queries (e.g., "recipes similar to carbonara")
    Complex,
    /// Semantic queries using synonyms/paraphrasing - ZERO lexical overlap (tests embedding-based retrieval)
    Semantic,
    /// Basic queries with partial keyword matching - SOME lexical overlap allowed
    Basic,
    /// Mix of all query types
    Mixed,
    /// Translate existing queries to a random target language (keeps qrels same)
    Translate,
}

impl QueryType {
    pub fn all_types() -> Vec<QueryType> {
        vec![
            QueryType::Natural,
            QueryType::Keyword,
            QueryType::Academic,
            QueryType::Complex,
            QueryType::Semantic,
            QueryType::Basic,
        ]
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            QueryType::Natural => "natural",
            QueryType::Keyword => "keyword",
            QueryType::Academic => "academic",
            QueryType::Complex => "complex",
            QueryType::Semantic => "semantic",
            QueryType::Basic => "basic",
            QueryType::Mixed => "mixed",
            QueryType::Translate => "translate",
        }
    }
}

impl std::fmt::Display for QueryType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

impl std::str::FromStr for QueryType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "natural" => Ok(QueryType::Natural),
            "keyword" => Ok(QueryType::Keyword),
            "academic" => Ok(QueryType::Academic),
            "complex" => Ok(QueryType::Complex),
            "semantic" => Ok(QueryType::Semantic),
            "basic" => Ok(QueryType::Basic),
            "mixed" => Ok(QueryType::Mixed),
            "translate" => Ok(QueryType::Translate),
            _ => Err(format!("Invalid query type: {}", s)),
        }
    }
}

/// Parse a comma-separated list of query types
pub fn parse_query_types(s: &str) -> Result<Vec<QueryType>, String> {
    let mut types = Vec::new();
    for part in s.split(',') {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            let qt: QueryType = trimmed.parse()?;
            if !types.contains(&qt) {
                types.push(qt);
            }
        }
    }
    if types.is_empty() {
        return Err("No valid query types specified".to_string());
    }
    Ok(types)
}

/// Main configuration for dataset generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Topic name (built-in or "llm-generated")
    pub topic: String,

    /// Path to existing corpus (if using user-provided documents)
    pub corpus_path: Option<PathBuf>,

    /// Number of documents to generate
    pub document_count: usize,

    /// Number of queries per query type
    pub queries_per_type: usize,

    /// Which query types to generate (if None, generate all)
    pub query_types: Option<Vec<QueryType>>,

    /// Output directory
    pub output_dir: PathBuf,

    /// LLM base URL
    pub base_url: String,

    /// LLM model identifier
    pub model: String,

    /// Whether to generate hard negatives
    pub generate_hard_negatives: bool,

    /// Batch size for checkpointing
    pub batch_size: usize,

    /// Number of concurrent LLM requests
    pub concurrency: usize,

    /// Scoring mode for relevance judgments
    pub scoring_mode: ScoringMode,

    /// Pool size for pooled scoring (top-k docs per query)
    pub pool_size: usize,

    /// Score scale (trec = 0-3, range = custom min/max)
    pub score_scale: ScoreScale,

    /// Minimum score for custom range (inclusive)
    pub score_min: u16,

    /// Maximum score for custom range (inclusive)
    pub score_max: u16,

    /// Document diversity mode
    pub doc_diversity: DocDiversity,

    /// Query diversity mode
    pub query_diversity: QueryDiversity,

    /// Similarity threshold for embedding-based deduplication (0.0-1.0)
    /// Higher = stricter (rejects more similar items)
    pub diversity_threshold: f32,

    /// Document categories for forced diversity (e.g., "soup,main,dessert")
    pub doc_categories: Option<Vec<String>>,

    /// Embedding API URL (for diversity checking, defaults to base_url)
    pub embedding_url: Option<String>,

    /// Embedding model (defaults to main model)
    pub embedding_model: Option<String>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            topic: "recipes".to_string(),
            corpus_path: None,
            document_count: 100,
            queries_per_type: 500,
            query_types: None,
            output_dir: PathBuf::from("./datasets"),
            base_url: "https://api.openai.com/v1".to_string(),
            model: "gpt-4".to_string(),
            generate_hard_negatives: true,
            batch_size: 50,
            concurrency: 1,
            scoring_mode: ScoringMode::Source,
            pool_size: 30,
            score_scale: ScoreScale::Trec,
            score_min: 0,
            score_max: 3,
            doc_diversity: DocDiversity::None,
            query_diversity: QueryDiversity::Exact,
            diversity_threshold: 0.85,
            doc_categories: None,
            embedding_url: None,
            embedding_model: None,
        }
    }
}

/// Runtime options (not persisted)
#[derive(Debug, Clone, Default)]
pub struct RuntimeOptions {
    /// Dry run mode - show what would happen without LLM calls
    pub dry_run: bool,

    /// Verbose output
    pub verbose: bool,

    /// Resume from checkpoint
    pub resume: bool,

    /// API key (not persisted)
    pub api_key: String,

    /// Skip creating merged and combined output directories
    pub no_merged: bool,
}
