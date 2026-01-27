use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::{Path, PathBuf};
use synthir::{
    benchmark::{run_benchmark, BenchmarkConfig},
    checkpoint::{CheckpointManager, GenerationPhase, ProgressState},
    config::{parse_query_types, GenerationConfig, OnExist, OutputMode, QueryType, RuntimeOptions, ScoreScale, ScoringMode},
    generation::{DocumentGenerator, QueryGenerator, RelevanceScorer},
    llm::{topic_generation_prompt, LLMProvider, LLMProviderConfig, MultiEndpointProvider, LanguageDetector, locale_to_language, EmbeddingClient},
    meta::{run_meta_generation, MetaConfig},
    mining::HardNegativeMiner,
    output::{
        read_corpus, read_qrels, write_qrels, write_queries, BeirQuery, DatasetMetadata, OutputOrganizer,
        QueryCounts, detect_dataset, DatasetFormat,
        write_ocr_queries, discover_datasets, read_ocr_corpus, QrelsSplit, read_ocr_queries,
    },
    topics::{create_custom_topic, get_builtin_topics, get_topic, TopicConfig},
    utils::{display_dry_run_info, display_topics, display_verbose_start},
};
use tracing::{info, warn, Level};
use tracing_subscriber::FmtSubscriber;

#[derive(Parser)]
#[command(name = "synthir")]
#[command(about = "Synthetic IR dataset generator for realistic information retrieval evaluation")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Generate a new dataset
    Generate {
        /// Topic name (built-in topic or "llm-generated")
        #[arg(short, long)]
        topic: Option<String>,

        /// Path to existing corpus.jsonl (skips document generation)
        #[arg(short, long)]
        corpus: Option<PathBuf>,

        /// Number of documents to generate
        #[arg(short, long, default_value = "100")]
        documents: usize,

        /// Number of queries per query type
        #[arg(short = 'q', long, default_value = "500")]
        queries_per_type: usize,

        /// Query types to generate (comma-separated: natural,keyword,academic,complex,semantic,mixed)
        /// If not specified, generates all types.
        #[arg(long, value_name = "TYPES")]
        query_types: Option<String>,

        /// Output directory
        #[arg(short, long, default_value = "./datasets")]
        output: PathBuf,

        /// LLM API base URL (comma-separated for multiple endpoints)
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key (or set OPENAI_API_KEY env var)
        #[arg(long)]
        api_key: Option<String>,

        /// Resume from checkpoint
        #[arg(long)]
        resume: bool,

        /// Dry run - show what would happen without LLM calls
        #[arg(long)]
        dry_run: bool,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Skip hard negative mining
        #[arg(long)]
        no_hard_negatives: bool,

        /// Skip creating merged and combined output directories
        #[arg(long)]
        no_merged: bool,

        /// Batch size for checkpointing
        #[arg(long, default_value = "50")]
        batch_size: usize,

        /// Number of concurrent LLM requests
        #[arg(short = 'j', long, default_value = "1")]
        concurrency: usize,

        /// Scoring mode: 'source', 'pooled', or 'exhaustive'
        #[arg(long, default_value = "source")]
        scoring_mode: String,

        /// Pool size for pooled scoring (top-k docs per query)
        #[arg(long, default_value = "30")]
        pool_size: usize,

        /// Score scale: 'trec' (0-3) or 'range' (custom min/max)
        #[arg(long, default_value = "trec")]
        score_scale: String,

        /// Minimum score for custom range (default: 0)
        #[arg(long, default_value = "0")]
        score_min: u16,

        /// Maximum score for custom range (default: 100)
        #[arg(long, default_value = "100")]
        score_max: u16,
    },

    /// Generate queries for an existing corpus
    Queries {
        /// Path to corpus.jsonl
        #[arg(short, long)]
        corpus: PathBuf,

        /// Query type to generate
        #[arg(short = 't', long, default_value = "natural")]
        query_type: String,

        /// Number of queries
        #[arg(short, long, default_value = "500")]
        count: usize,

        /// Output directory
        #[arg(short, long)]
        output: PathBuf,

        /// LLM API base URL
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key
        #[arg(long)]
        api_key: Option<String>,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,
    },

    /// Benchmark LLM endpoint to find optimal concurrency
    Benchmark {
        /// LLM API base URL
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key
        #[arg(long)]
        api_key: Option<String>,

        /// Number of samples per concurrency level
        #[arg(short, long, default_value = "10")]
        samples: usize,

        /// Concurrency levels to test (comma-separated)
        #[arg(long, default_value = "1,2,4,8,16,32")]
        levels: String,
    },

    /// Generate multiple datasets automatically
    Meta {
        /// Number of topics to generate
        #[arg(short = 't', long, default_value = "5")]
        topics: usize,

        /// Number of documents per topic
        #[arg(short, long, default_value = "100")]
        documents: usize,

        /// Number of queries per query type
        #[arg(short = 'q', long, default_value = "500")]
        queries_per_type: usize,

        /// Output directory
        #[arg(short, long, default_value = "./multi-datasets")]
        output: PathBuf,

        /// LLM API base URL
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key
        #[arg(long)]
        api_key: Option<String>,

        /// Use shared corpus across all topics
        #[arg(long)]
        shared_corpus: bool,

        /// Skip hard negative mining
        #[arg(long)]
        no_hard_negatives: bool,

        /// Number of concurrent LLM requests
        #[arg(short = 'j', long, default_value = "1")]
        concurrency: usize,

        /// Batch size for checkpointing
        #[arg(long, default_value = "50")]
        batch_size: usize,
    },

    /// List available topics
    Topics,

    /// Remix an existing dataset with new queries (clone corpus, replace queries/qrels)
    Remix {
        /// Path to source dataset directory (auto-detects BEIR or OCR format)
        #[arg(short, long)]
        source: PathBuf,

        /// Name for the output dataset (e.g., "semantic_nfcorpus"). Required unless --in-place is used.
        #[arg(short = 'n', long)]
        output_name: Option<String>,

        /// Output directory (dataset will be created as subdirectory). Defaults to source's parent directory.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Replace queries/qrels in-place (modifies source dataset directly)
        #[arg(long)]
        in_place: bool,

        /// Query types to generate (comma-separated: natural,keyword,academic,complex,semantic,basic,mixed)
        #[arg(long, value_name = "TYPES", default_value = "semantic")]
        query_types: String,

        /// Number of queries per split (if source has train/dev/test, preserves ratio)
        #[arg(short = 'q', long)]
        queries_per_type: Option<usize>,

        /// LLM API base URL
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key (or set OPENAI_API_KEY env var)
        #[arg(long)]
        api_key: Option<String>,

        /// Number of concurrent LLM requests
        #[arg(short = 'j', long, default_value = "1")]
        concurrency: usize,

        /// Scoring mode: source (1-to-1), pooled (BM25 candidates), exhaustive (all pairs)
        #[arg(long, default_value = "source")]
        scoring_mode: String,

        /// Pool size for pooled scoring (top-k docs per query)
        #[arg(long, default_value = "30")]
        pool_size: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Dry run - show what would happen without LLM calls
        #[arg(long)]
        dry_run: bool,

        /// Trust locale directory names (e.g., fr-fr → French) instead of detecting language with LLM
        #[arg(long)]
        trust_locale: bool,
    },

    /// Batch remix all compatible datasets in a directory
    RemixBatch {
        /// Directory containing multiple datasets to remix
        #[arg(short, long)]
        source: PathBuf,

        /// Output mode: sibling (next to originals) or grouped (in separate directory)
        #[arg(long, default_value = "sibling")]
        output_mode: String,

        /// Output directory for grouped mode (defaults to <source>-remixed)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Behavior when output already exists: skip, overwrite, or ask
        #[arg(long, default_value = "skip")]
        on_exist: String,

        /// Query types to generate (comma-separated: natural,keyword,academic,complex,semantic,basic,mixed)
        #[arg(long, value_name = "TYPES", default_value = "semantic")]
        query_types: String,

        /// Number of queries per type (auto-derived from source if not specified)
        #[arg(short = 'q', long)]
        queries_per_type: Option<usize>,

        /// LLM API base URL
        #[arg(long, default_value = "https://api.openai.com/v1")]
        base_url: String,

        /// Model identifier
        #[arg(short, long, default_value = "gpt-4")]
        model: String,

        /// API key (or set OPENAI_API_KEY env var)
        #[arg(long)]
        api_key: Option<String>,

        /// Number of concurrent LLM requests
        #[arg(short = 'j', long, default_value = "1")]
        concurrency: usize,

        /// Scoring mode: source (1-to-1), pooled (BM25 candidates), exhaustive (all pairs)
        #[arg(long, default_value = "source")]
        scoring_mode: String,

        /// Pool size for pooled scoring (top-k docs per query)
        #[arg(long, default_value = "30")]
        pool_size: usize,

        /// Verbose output
        #[arg(short, long)]
        verbose: bool,

        /// Dry run - show what would happen without LLM calls
        #[arg(long)]
        dry_run: bool,

        /// Trust locale directory names (e.g., fr-fr → French) instead of detecting language with LLM
        #[arg(long)]
        trust_locale: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Generate {
            topic,
            corpus,
            documents,
            queries_per_type,
            query_types,
            output,
            base_url,
            model,
            api_key,
            resume,
            dry_run,
            verbose,
            no_hard_negatives,
            no_merged,
            batch_size,
            concurrency,
            scoring_mode,
            pool_size,
            score_scale,
            score_min,
            score_max,
        } => {
            // Set up logging
            let level = if verbose { Level::DEBUG } else { Level::INFO };
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)?;

            // Get API key
            let api_key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .context("API key required: use --api-key or set OPENAI_API_KEY")?;

            // Parse scoring mode
            let scoring_mode: ScoringMode = scoring_mode
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;

            // Parse score scale
            let score_scale: ScoreScale = score_scale
                .parse()
                .map_err(|e: String| anyhow::anyhow!(e))?;

            // Validate score range
            if score_min > score_max {
                anyhow::bail!("--score-min ({}) cannot be greater than --score-max ({})", score_min, score_max);
            }

            // Parse query types if specified
            let selected_query_types = query_types
                .map(|s| parse_query_types(&s))
                .transpose()
                .map_err(|e| anyhow::anyhow!("Invalid query types: {}", e))?;

            // Determine topic: use provided topic, or derive from output directory name when using existing corpus
            let effective_topic = topic.clone().unwrap_or_else(|| {
                if corpus.is_some() {
                    // When using existing corpus, use output directory name as topic (no subdirectory)
                    output
                        .file_name()
                        .and_then(|n| n.to_str())
                        .map(|s| s.to_string())
                        .unwrap_or_else(|| "dataset".to_string())
                } else {
                    "recipes".to_string()
                }
            });

            // Build config
            let config = GenerationConfig {
                topic: effective_topic,
                corpus_path: corpus.clone(),
                document_count: documents,
                queries_per_type,
                query_types: selected_query_types,
                output_dir: output.clone(),
                base_url: base_url.clone(),
                model: model.clone(),
                generate_hard_negatives: !no_hard_negatives,
                batch_size,
                concurrency,
                scoring_mode,
                pool_size,
                score_scale,
                score_min,
                score_max,
            };

            let options = RuntimeOptions {
                dry_run,
                verbose,
                resume,
                api_key: api_key.clone(),
                no_merged,
            };

            // Get or generate topic config
            let topic_config = get_or_generate_topic(&config, &options).await?;

            // Dry run mode
            if dry_run {
                display_dry_run_info(&config, &topic_config, &options);
                return Ok(());
            }

            if verbose {
                display_verbose_start(&config, &topic_config);
            }

            // Run generation
            run_generation(config, topic_config, options).await?;

            info!("Dataset generation complete!");
        }

        Commands::Queries {
            corpus,
            query_type,
            count,
            output,
            base_url,
            model,
            api_key,
            verbose,
        } => {
            let level = if verbose { Level::DEBUG } else { Level::INFO };
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)?;

            let api_key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .context("API key required")?;

            let query_type: QueryType = query_type.parse().map_err(|e: String| anyhow::anyhow!(e))?;

            run_query_generation(corpus, query_type, count, output, base_url, model, api_key)
                .await?;
        }

        Commands::Benchmark {
            base_url,
            model,
            api_key,
            samples,
            levels,
        } => {
            let api_key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .context("API key required: use --api-key or set OPENAI_API_KEY")?;

            let concurrency_levels: Vec<usize> = levels
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();

            if concurrency_levels.is_empty() {
                anyhow::bail!("Invalid concurrency levels: {}", levels);
            }

            let config = BenchmarkConfig {
                base_url,
                api_key,
                model,
                concurrency_levels,
                samples_per_level: samples,
            };

            run_benchmark(config).await?;
        }

        Commands::Meta {
            topics,
            documents,
            queries_per_type,
            output,
            base_url,
            model,
            api_key,
            shared_corpus,
            no_hard_negatives,
            concurrency,
            batch_size,
        } => {
            // Set up logging
            let subscriber = FmtSubscriber::builder()
                .with_max_level(Level::INFO)
                .finish();
            tracing::subscriber::set_global_default(subscriber)?;

            let api_key = api_key
                .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                .context("API key required: use --api-key or set OPENAI_API_KEY")?;

            let config = MetaConfig {
                topic_count: topics,
                document_count: documents,
                queries_per_type,
                output_dir: output,
                base_url,
                model,
                api_key,
                shared_corpus,
                generate_hard_negatives: !no_hard_negatives,
                concurrency,
                batch_size,
            };

            run_meta_generation(config).await?;
        }

        Commands::Topics => {
            let topics = get_builtin_topics();
            let mut topic_list: Vec<_> = topics
                .iter()
                .map(|(name, config)| (name.clone(), config.description.clone()))
                .collect();
            topic_list.sort_by(|a, b| a.0.cmp(&b.0));
            display_topics(&topic_list);
        }

        Commands::Remix {
            source,
            output_name,
            output,
            in_place,
            query_types,
            queries_per_type,
            base_url,
            model,
            api_key,
            concurrency,
            scoring_mode,
            pool_size,
            verbose,
            dry_run,
            trust_locale,
        } => {
            let level = if verbose { Level::DEBUG } else { Level::INFO };
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)?;

            // Validate: must specify either --output-name or --in-place
            if output_name.is_none() && !in_place {
                anyhow::bail!("Must specify either --output-name or --in-place");
            }

            // API key only required for non-dry-run
            let api_key = if dry_run {
                api_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .unwrap_or_default()
            } else {
                api_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .context("API key required: use --api-key or set OPENAI_API_KEY")?
            };

            let selected_query_types = parse_query_types(&query_types)
                .map_err(|e| anyhow::anyhow!("Invalid query types: {}", e))?;

            let scoring: ScoringMode = scoring_mode.parse()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            run_remix(
                source,
                output_name,
                output,
                in_place,
                selected_query_types,
                queries_per_type,
                base_url,
                model,
                api_key,
                concurrency,
                scoring,
                pool_size,
                dry_run,
                trust_locale,
            )
            .await?;

            if !dry_run {
                info!("Dataset remix complete!");
            }
        }

        Commands::RemixBatch {
            source,
            output_mode,
            output,
            on_exist,
            query_types,
            queries_per_type,
            base_url,
            model,
            api_key,
            concurrency,
            scoring_mode,
            pool_size,
            verbose,
            dry_run,
            trust_locale,
        } => {
            let level = if verbose { Level::DEBUG } else { Level::INFO };
            let subscriber = FmtSubscriber::builder().with_max_level(level).finish();
            tracing::subscriber::set_global_default(subscriber)?;

            // API key only required for non-dry-run
            let api_key = if dry_run {
                api_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .unwrap_or_default()
            } else {
                api_key
                    .or_else(|| std::env::var("OPENAI_API_KEY").ok())
                    .context("API key required: use --api-key or set OPENAI_API_KEY")?
            };

            let selected_query_types = parse_query_types(&query_types)
                .map_err(|e| anyhow::anyhow!("Invalid query types: {}", e))?;

            let scoring: ScoringMode = scoring_mode
                .parse()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let output_mode: OutputMode = output_mode
                .parse()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            let on_exist: OnExist = on_exist
                .parse()
                .map_err(|e| anyhow::anyhow!("{}", e))?;

            run_remix_batch(
                source,
                output_mode,
                output,
                on_exist,
                selected_query_types,
                queries_per_type,
                base_url,
                model,
                api_key,
                concurrency,
                scoring,
                pool_size,
                dry_run,
                trust_locale,
            )
            .await?;
        }
    }

    Ok(())
}

async fn get_or_generate_topic(
    config: &GenerationConfig,
    options: &RuntimeOptions,
) -> Result<TopicConfig> {
    // When using existing corpus, accept any topic name (no document generation needed)
    if config.corpus_path.is_some() {
        return Ok(create_custom_topic(
            config.topic.clone(),
            format!("Existing corpus: {}", config.topic),
        ));
    }

    if config.topic == "llm-generated" {
        info!("Generating random topic with LLM...");

        let provider = LLMProvider::new(LLMProviderConfig {
            base_url: config.base_url.clone(),
            api_key: options.api_key.clone(),
            model: config.model.clone(),
            max_retries: 3,
        })?;

        let topic_name = provider.generate_topic(topic_generation_prompt()).await?;
        info!("Generated topic: {}", topic_name);

        Ok(create_custom_topic(
            topic_name.clone(),
            format!("LLM-generated topic: {}", topic_name),
        ))
    } else if let Some(topic_config) = get_topic(&config.topic) {
        Ok(topic_config)
    } else {
        anyhow::bail!(
            "Unknown topic: {}. Use 'synthir topics' to list available topics.",
            config.topic
        );
    }
}

async fn run_generation(
    config: GenerationConfig,
    topic_config: TopicConfig,
    options: RuntimeOptions,
) -> Result<()> {
    // Create provider - supports multiple URLs or multiple model identifiers
    let multi_provider = if config.model.contains(',') {
        // Multiple model identifiers (for manually loaded LMStudio instances)
        MultiEndpointProvider::new_multi_model(
            &config.base_url,
            &options.api_key,
            &config.model,
            3,
        )?
    } else if config.base_url.contains(',') {
        // Multiple endpoints (for multiple LMStudio servers on different ports)
        MultiEndpointProvider::new(
            &config.base_url,
            &options.api_key,
            &config.model,
            3,
        )?
    } else {
        // Single endpoint, single model
        MultiEndpointProvider::new(
            &config.base_url,
            &options.api_key,
            &config.model,
            3,
        )?
    };

    if multi_provider.endpoint_count() > 1 {
        info!(
            "Using {} endpoints/instances for parallel requests",
            multi_provider.endpoint_count()
        );
    }

    // Get first provider for generators that need single provider reference
    let provider = multi_provider.first();

    // Use flat output (no topic subdirectory) when using existing corpus
    let organizer = if config.corpus_path.is_some() {
        OutputOrganizer::new_flat(config.output_dir.clone(), topic_config.name.clone())
    } else {
        OutputOrganizer::new(config.output_dir.clone(), topic_config.name.clone())
    };

    // Create directory structure for only the selected query types
    let types_to_create: Vec<QueryType> = config
        .query_types
        .clone()
        .unwrap_or_else(|| {
            let mut all = QueryType::all_types();
            all.push(QueryType::Mixed);
            all
        });
    organizer.create_structure_for_types(&types_to_create, config.generate_hard_negatives, !options.no_merged)?;

    // Load or create progress state
    let mut state = if options.resume {
        ProgressState::load(&organizer.topic_dir())?
            .unwrap_or_else(|| ProgressState::new(config.clone()))
    } else {
        ProgressState::new(config.clone())
    };

    let mut checkpoint_mgr = CheckpointManager::new(organizer.topic_dir(), config.batch_size);

    // Phase 1: Document Generation
    let documents = if let Some(corpus_path) = &config.corpus_path {
        info!("Loading existing corpus from {}", corpus_path.display());
        read_corpus(corpus_path)?
    } else if state.phase == GenerationPhase::DocumentGeneration
        || state.completed_documents.is_empty()
    {
        info!(
            "Generating {} documents (concurrency: {})...",
            config.document_count, config.concurrency
        );
        let doc_gen = DocumentGenerator::new(&provider, topic_config.clone(), options.dry_run);
        let docs = doc_gen
            .generate_all_concurrent(
                config.document_count,
                &organizer.corpus_path(),
                &mut state,
                &mut checkpoint_mgr,
                config.concurrency,
            )
            .await?;

        state.advance_phase();
        checkpoint_mgr.force_save(&state)?;
        docs
    } else {
        info!("Loading documents from checkpoint...");
        read_corpus(&organizer.corpus_path())?
    };

    // Phase 2: Query Generation
    if state.phase == GenerationPhase::QueryGeneration
        || state.phase == GenerationPhase::DocumentGeneration
    {
        state.phase = GenerationPhase::QueryGeneration;

        info!(
            "Generating queries (concurrency: {})...",
            config.concurrency
        );
        let query_gen = QueryGenerator::new(&provider, options.dry_run);

        let mut all_query_doc_pairs: Vec<(BeirQuery, String)> = Vec::new();

        // Determine which query types to generate
        let types_to_generate: Vec<QueryType> = config
            .query_types
            .clone()
            .unwrap_or_else(QueryType::all_types);

        // Generate each selected query type (excluding Mixed, handled separately)
        for query_type in types_to_generate.iter().filter(|t| **t != QueryType::Mixed) {
            let pairs = query_gen
                .generate_for_type_concurrent(
                    &documents,
                    *query_type,
                    config.queries_per_type,
                    &organizer.queries_path(*query_type),
                    &mut state,
                    &mut checkpoint_mgr,
                    config.concurrency,
                )
                .await?;
            all_query_doc_pairs.extend(pairs);
        }

        // Generate mixed queries only if Mixed is in the selected types (or all types selected)
        let generate_mixed = config
            .query_types
            .as_ref()
            .map(|types| types.contains(&QueryType::Mixed))
            .unwrap_or(true);

        if generate_mixed {
            let _mixed_pairs = query_gen
                .generate_mixed(
                    &documents,
                    config.queries_per_type,
                    &organizer.queries_path(QueryType::Mixed),
                    &mut state,
                    &mut checkpoint_mgr,
                )
                .await?;
        }

        state.advance_phase();
        checkpoint_mgr.force_save(&state)?;

        // Phase 3: Relevance Scoring
        info!(
            "Scoring relevance (mode: {}, scale: {}, concurrency: {})...",
            config.scoring_mode, config.score_scale, config.concurrency
        );
        let scorer = RelevanceScorer::with_scale(
            &provider,
            options.dry_run,
            config.score_scale,
            config.score_min,
            config.score_max,
        );

        // Check if we're in semantic-only mode
        let is_semantic_only = types_to_generate.iter().all(|qt| *qt == QueryType::Semantic);

        let qrels = match &config.scoring_mode {
            ScoringMode::Source => {
                scorer
                    .score_source_pairs_concurrent(&all_query_doc_pairs, &documents, config.concurrency)
                    .await?
            }
            ScoringMode::Pooled => {
                // Collect all queries for pooled scoring
                let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
                if is_semantic_only {
                    // Use embedding-based pooling for semantic queries
                    info!("Semantic queries detected: using embedding-based pooling (HNSW)");
                    let embedding_client = EmbeddingClient::new(&config.base_url, &config.model);
                    scorer
                        .score_pooled_semantic(&all_queries, &documents, &embedding_client, config.pool_size, config.concurrency)
                        .await?
                } else {
                    scorer
                        .score_pooled_with_options(&all_queries, &documents, config.pool_size, config.concurrency, false)
                        .await?
                }
            }
            ScoringMode::Exhaustive => {
                // Score every query against every document
                let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
                let qrels = scorer
                    .score_exhaustive(&all_queries, &documents, config.concurrency)
                    .await?;
                // For semantic mode, filter out qrels where query has word overlap with doc
                if is_semantic_only {
                    filter_semantic_qrels(qrels, &all_queries, &documents)
                } else {
                    qrels
                }
            }
        };

        // Write qrels for each generated query type
        for query_type in types_to_generate.iter().filter(|t| **t != QueryType::Mixed) {
            let type_qrels: Vec<_> = qrels
                .iter()
                .filter(|q| q.query_id.starts_with(query_type.as_str()))
                .cloned()
                .collect();
            write_qrels(&organizer.qrels_path(*query_type), &type_qrels)?;
        }

        state.advance_phase();
        checkpoint_mgr.force_save(&state)?;

        // Phase 4: Hard Negative Mining (optional, requires merged output)
        if config.generate_hard_negatives && !options.no_merged {
            info!("Mining hard negatives...");
            let miner = HardNegativeMiner::new(&provider, options.dry_run);

            // Collect all queries
            let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();

            let hard_negatives = miner
                .mine_hard_negatives(&all_queries, &documents, &qrels, 3)
                .await?;

            // Merge qrels with hard negatives
            let mut merged_qrels = qrels.clone();
            merged_qrels.extend(hard_negatives);

            // Write merged outputs
            write_qrels(
                &organizer.hard_negatives_merged_dir().join("qrels.tsv"),
                &merged_qrels,
            )?;

            // Copy queries to merged directory
            let all_query_refs: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
            write_queries(
                &organizer.hard_negatives_merged_dir().join("queries.jsonl"),
                &all_query_refs,
            )?;
        }

        // Write merged and combined folders only if not disabled
        if !options.no_merged {
            // Write general merged (without hard negatives)
            let all_query_refs: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
            write_queries(
                &organizer.general_merged_dir().join("queries.jsonl"),
                &all_query_refs,
            )?;
            write_qrels(&organizer.general_merged_dir().join("qrels.tsv"), &qrels)?;

            // Write combined folder (corpus + all queries + all qrels in one place)
            info!("Writing combined output folder...");
            // Use original corpus path if provided, otherwise use organizer's corpus path
            let corpus_source = config.corpus_path.as_ref()
                .map(|p| p.clone())
                .unwrap_or_else(|| organizer.corpus_path());
            std::fs::copy(
                &corpus_source,
                organizer.combined_dir().join("corpus.jsonl"),
            )?;
            write_queries(
                &organizer.combined_dir().join("queries.jsonl"),
                &all_query_refs,
            )?;
            write_qrels(&organizer.combined_dir().join("qrels.tsv"), &qrels)?;
        }

        state.advance_phase();
        checkpoint_mgr.force_save(&state)?;
    }

    // Write metadata
    let all_types_with_mixed = {
        let mut types = QueryType::all_types();
        types.push(QueryType::Mixed);
        types
    };
    let generated_types = config.query_types.clone().unwrap_or(all_types_with_mixed);

    let metadata = DatasetMetadata {
        topic: topic_config.name.clone(),
        document_count: documents.len(),
        query_counts: QueryCounts {
            natural: if generated_types.contains(&QueryType::Natural) { config.queries_per_type } else { 0 },
            keyword: if generated_types.contains(&QueryType::Keyword) { config.queries_per_type } else { 0 },
            academic: if generated_types.contains(&QueryType::Academic) { config.queries_per_type } else { 0 },
            complex: if generated_types.contains(&QueryType::Complex) { config.queries_per_type } else { 0 },
            semantic: if generated_types.contains(&QueryType::Semantic) { config.queries_per_type } else { 0 },
            basic: if generated_types.contains(&QueryType::Basic) { config.queries_per_type } else { 0 },
            mixed: if generated_types.contains(&QueryType::Mixed) { config.queries_per_type } else { 0 },
        },
        generation_timestamp: chrono_lite_timestamp(),
        model_used: config.model.clone(),
        has_hard_negatives: config.generate_hard_negatives,
    };
    organizer.write_metadata(&metadata)?;

    state.phase = GenerationPhase::Complete;
    checkpoint_mgr.force_save(&state)?;

    Ok(())
}

async fn run_query_generation(
    corpus_path: PathBuf,
    query_type: QueryType,
    count: usize,
    output: PathBuf,
    base_url: String,
    model: String,
    api_key: String,
) -> Result<()> {
    let provider = LLMProvider::new(LLMProviderConfig {
        base_url,
        api_key: api_key.clone(),
        model,
        max_retries: 3,
    })?;

    let documents = read_corpus(&corpus_path)?;
    info!("Loaded {} documents from corpus", documents.len());

    let config = GenerationConfig {
        queries_per_type: count,
        output_dir: output.clone(),
        ..Default::default()
    };

    let mut state = ProgressState::new(config.clone());
    let mut checkpoint_mgr = CheckpointManager::new(output.clone(), 50);

    let query_gen = QueryGenerator::new(&provider, false);

    std::fs::create_dir_all(&output)?;
    let queries_path = output.join("queries.jsonl");

    let pairs = query_gen
        .generate_for_type(
            &documents,
            query_type,
            count,
            &queries_path,
            &mut state,
            &mut checkpoint_mgr,
        )
        .await?;

    // Score relevance
    let scorer = RelevanceScorer::new(&provider, false);
    let qrels = scorer.score_source_pairs(&pairs, &documents).await?;

    write_qrels(&output.join("qrels.tsv"), &qrels)?;

    info!(
        "Generated {} queries and {} relevance judgments",
        pairs.len(),
        qrels.len()
    );

    Ok(())
}

/// Simple timestamp without heavy chrono dependency
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

/// Minimum word length for overlap checking
const MIN_OVERLAP_WORD_LENGTH: usize = 3;

/// Check if query has word overlap with document text
fn has_word_overlap(query: &str, document: &str) -> bool {
    use std::collections::HashSet;

    // Normalize and tokenize document
    let doc_words: HashSet<String> = document
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MIN_OVERLAP_WORD_LENGTH)
        .map(|w| w.to_string())
        .collect();

    // Tokenize query
    let query_lower = query.to_lowercase();
    let query_words: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MIN_OVERLAP_WORD_LENGTH)
        .map(|w| w.to_string())
        .collect();

    for word in &query_words {
        // Check exact match
        if doc_words.contains(word) {
            return true;
        }
        // Check stem overlap (first 4-6 chars)
        let word_chars: Vec<char> = word.chars().collect();
        if word_chars.len() >= 4 {
            for doc_word in &doc_words {
                let doc_chars: Vec<char> = doc_word.chars().collect();
                if doc_chars.len() >= 4 {
                    let min_len = word_chars.len().min(doc_chars.len()).min(6);
                    let word_prefix: String = word_chars[..min_len].iter().collect();
                    let doc_prefix: String = doc_chars[..min_len].iter().collect();
                    if word_prefix == doc_prefix {
                        return true;
                    }
                }
            }
        }
    }

    false
}

/// Filter qrels to only include pairs with zero word overlap (for semantic queries)
fn filter_semantic_qrels(
    qrels: Vec<synthir::output::Qrel>,
    queries: &[BeirQuery],
    documents: &[synthir::output::BeirDocument],
) -> Vec<synthir::output::Qrel> {
    use std::collections::HashMap;

    let query_map: HashMap<&str, &BeirQuery> = queries.iter().map(|q| (q.id.as_str(), q)).collect();
    let doc_map: HashMap<&str, &synthir::output::BeirDocument> = documents.iter().map(|d| (d.id.as_str(), d)).collect();

    let before_count = qrels.len();
    let filtered: Vec<_> = qrels
        .into_iter()
        .filter(|qrel| {
            if let (Some(query), Some(doc)) = (query_map.get(qrel.query_id.as_str()), doc_map.get(qrel.doc_id.as_str())) {
                let full_doc_text = format!("{} {}", doc.title, doc.text);
                !has_word_overlap(&query.text, &full_doc_text)
            } else {
                false
            }
        })
        .collect();

    let filtered_count = before_count - filtered.len();
    if filtered_count > 0 {
        info!("Filtered {} qrels with word overlap (semantic mode), {} remaining", filtered_count, filtered.len());
    }

    filtered
}

/// Remix an existing dataset with new queries
async fn run_remix(
    source: PathBuf,
    output_name: Option<String>,
    output_dir: Option<PathBuf>,
    in_place: bool,
    query_types: Vec<QueryType>,
    queries_per_type: Option<usize>,
    base_url: String,
    model: String,
    api_key: String,
    concurrency: usize,
    scoring_mode: ScoringMode,
    pool_size: usize,
    dry_run: bool,
    trust_locale: bool,
) -> Result<()> {
    // Detect source dataset format
    info!("Detecting dataset format for {}...", source.display());
    let dataset_info = detect_dataset(&source)?;

    // Determine which locales to process
    let locales_to_process: Vec<Option<String>> = if dataset_info.all_locales.is_empty() {
        // No locale subdirectories - process root
        vec![None]
    } else {
        // Process all locales
        dataset_info
            .all_locales
            .iter()
            .map(|l| Some(l.clone()))
            .collect()
    };

    info!(
        "Detected {} format{}",
        dataset_info.format,
        if locales_to_process.len() > 1 {
            format!(" ({} locales: {})", locales_to_process.len(), dataset_info.all_locales.join(", "))
        } else if let Some(Some(l)) = locales_to_process.first() {
            format!(" (locale: {})", l)
        } else {
            String::new()
        }
    );

    // Determine base destination directory (without locale)
    let dest_dir = if in_place {
        source.clone()
    } else {
        let name = output_name.as_ref().unwrap();
        let base_dir = output_dir.unwrap_or_else(|| {
            source.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from("."))
        });
        base_dir.join(name)
    };

    // Save base_url for embedding client (before moving into provider config)
    let embedding_base_url = base_url.clone();

    // Create LLM provider once (shared across all locales)
    let provider = LLMProvider::new(LLMProviderConfig {
        base_url,
        api_key,
        model: model.clone(),
        max_retries: 3,
    })?;

    // Process each locale
    for (locale_idx, locale) in locales_to_process.iter().enumerate() {
        let locale_label = locale.as_deref().unwrap_or("root");

        if locales_to_process.len() > 1 {
            info!(
                "\n[Locale {}/{}] Processing {}...",
                locale_idx + 1,
                locales_to_process.len(),
                locale_label
            );
        }

        // Determine paths for this locale
        let (source_locale_dir, effective_dest) = if let Some(l) = locale {
            (source.join(l), dest_dir.join(l))
        } else {
            (source.clone(), dest_dir.clone())
        };

        // Get corpus path for this locale
        let corpus_path = match dataset_info.format {
            DatasetFormat::Beir => source_locale_dir.join("corpus.jsonl"),
            DatasetFormat::Ocr => source_locale_dir.join("label.json"),
        };

        if !corpus_path.exists() {
            warn!("Corpus not found at {}, skipping locale", corpus_path.display());
            continue;
        }

        // Read corpus for this locale
        let documents = match dataset_info.format {
            DatasetFormat::Beir => read_corpus(&corpus_path)?,
            DatasetFormat::Ocr => read_ocr_corpus(&corpus_path)?,
        };
        info!("Loaded {} documents for {}", documents.len(), locale_label);

        // Detect language for this locale
        let detected_language: Option<String> = if trust_locale {
            // Trust locale name if present (e.g., fr-fr → French)
            if let Some(l) = locale {
                let lang = locale_to_language(l);
                if let Some(ref lang_name) = lang {
                    info!("Using language from locale: {} → {}", l, lang_name);
                }
                lang
            } else {
                None
            }
        } else if !dry_run {
            // Detect language from existing queries or documents using LLM
            let detector = LanguageDetector::new(&provider);

            // Try to get text samples - prefer existing queries, fallback to documents
            let queries_path = source_locale_dir.join(
                if dataset_info.format == DatasetFormat::Ocr { "queries.json" } else { "queries.jsonl" }
            );

            let samples: Vec<String> = if queries_path.exists() {
                // Try to read existing queries
                match dataset_info.format {
                    DatasetFormat::Ocr => {
                        read_ocr_queries(&queries_path)
                            .map(|(qs, _)| qs.into_iter().map(|q| q.text).collect())
                            .unwrap_or_default()
                    }
                    DatasetFormat::Beir => {
                        synthir::output::read_queries(&queries_path)
                            .map(|qs| qs.into_iter().map(|q| q.text).collect())
                            .unwrap_or_default()
                    }
                }
            } else {
                Vec::new()
            };

            // If no queries, use document text
            let samples = if samples.is_empty() {
                documents.iter().take(30).map(|d| d.text.clone()).collect()
            } else {
                samples
            };

            if samples.is_empty() {
                None
            } else {
                // Progressive retry with increasing sample sizes
                match detector.detect_with_retry(&samples, &[5, 15, 30]).await {
                    Ok(Some(result)) => Some(result.language),
                    Ok(None) => {
                        warn!("Language detection inconclusive, defaulting to English");
                        Some("English".to_string())
                    }
                    Err(e) => {
                        warn!("Language detection failed: {}, defaulting to English", e);
                        Some("English".to_string())
                    }
                }
            }
        } else {
            None
        };

        // Analyze qrels for this locale (if present)
        let locale_qrels_paths = find_qrels_files_in_dir(&source_locale_dir);
        let splits: Vec<QrelsSplit> = if dataset_info.format == DatasetFormat::Ocr {
            let queries_path = source_locale_dir.join("queries.json");
            if queries_path.exists() {
                let (queries, _) = read_ocr_queries(&queries_path)?;
                vec![QrelsSplit {
                    name: "queries".to_string(),
                    path: queries_path,
                    count: queries.len(),
                }]
            } else {
                vec![]
            }
        } else {
            let mut splits = Vec::new();
            for qrels_path in &locale_qrels_paths {
                let qrels = read_qrels(qrels_path)?;
                let name = qrels_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("qrels")
                    .to_string();
                splits.push(QrelsSplit {
                    name,
                    path: qrels_path.clone(),
                    count: qrels.len(),
                });
            }
            splits
        };

        let total_qrels: usize = splits.iter().map(|s| s.count).sum();

        // Determine queries per type
        let queries_count = queries_per_type.unwrap_or_else(|| {
            let per_type = total_qrels / query_types.len().max(1);
            per_type.max(10)
        });

        if dry_run {
            if in_place {
                info!("DRY RUN - Would modify in-place: {}", effective_dest.display());
            } else {
                info!("DRY RUN - Would create: {}", effective_dest.display());
            }
            info!("  Corpus: {} documents", documents.len());
            info!("  Query types: {:?}", query_types);
            info!("  Queries per type: {}", queries_count);
            if let Some(ref lang) = detected_language {
                info!("  Language: {}", lang);
            }
            continue;
        }

        // For clone mode, create directory and copy corpus
        if !in_place {
            std::fs::create_dir_all(&effective_dest)?;

            let dest_corpus_path = match dataset_info.format {
                DatasetFormat::Beir => effective_dest.join("corpus.jsonl"),
                DatasetFormat::Ocr => effective_dest.join("label.json"),
            };
            std::fs::copy(&corpus_path, &dest_corpus_path)?;
            info!("Copied corpus to {}", dest_corpus_path.display());

            // Copy images directory for OCR format if it exists
            if dataset_info.format == DatasetFormat::Ocr {
                let source_images = source_locale_dir.join("images");
                if source_images.exists() {
                    let dest_images = effective_dest.join("images");
                    copy_dir_recursive(&source_images, &dest_images)?;
                    info!("Copied images directory");
                }
            }
        }

        // Generate queries
        let query_gen = QueryGenerator::new(&provider, false).with_language(detected_language.clone());
        let mut all_queries: Vec<BeirQuery> = Vec::new();
        let mut all_query_doc_pairs: Vec<(BeirQuery, String)> = Vec::new();

        let config = GenerationConfig {
            queries_per_type: queries_count,
            output_dir: effective_dest.clone(),
            concurrency,
            ..Default::default()
        };
        let mut state = ProgressState::new(config.clone());
        let mut checkpoint_mgr = CheckpointManager::new(effective_dest.clone(), 50);

        for query_type in &query_types {
            if *query_type == QueryType::Mixed {
                continue;
            }

            info!("Generating {} queries of type '{}'...", queries_count, query_type);

            let temp_queries_path = effective_dest.join(format!("temp_{}_queries.jsonl", query_type.as_str()));

            let pairs = query_gen
                .generate_for_type_concurrent(
                    &documents,
                    *query_type,
                    queries_count,
                    &temp_queries_path,
                    &mut state,
                    &mut checkpoint_mgr,
                    concurrency,
                )
                .await?;

            all_queries.extend(pairs.iter().map(|(q, _)| q.clone()));
            all_query_doc_pairs.extend(pairs);

            let _ = std::fs::remove_file(&temp_queries_path);
        }

        // Handle mixed queries if requested
        if query_types.contains(&QueryType::Mixed) {
            info!("Generating {} mixed queries...", queries_count);
            let temp_mixed_path = effective_dest.join("temp_mixed_queries.jsonl");

            let mixed_pairs = query_gen
                .generate_mixed(
                    &documents,
                    queries_count,
                    &temp_mixed_path,
                    &mut state,
                    &mut checkpoint_mgr,
                )
                .await?;

            let pairs: Vec<(BeirQuery, String)> = mixed_pairs
                .iter()
                .map(|(q, doc_id, _)| (q.clone(), doc_id.clone()))
                .collect();

            all_queries.extend(pairs.iter().map(|(q, _)| q.clone()));
            all_query_doc_pairs.extend(pairs);

            let _ = std::fs::remove_file(&temp_mixed_path);
        }

        // Score relevance
        info!("Scoring relevance (mode: {}, concurrency: {})...", scoring_mode, concurrency);
        let scorer = RelevanceScorer::new(&provider, false);

        // Check if we're in semantic mode (all query types are semantic)
        let is_semantic_only = query_types.iter().all(|qt| *qt == QueryType::Semantic);

        let all_qrels = match &scoring_mode {
            ScoringMode::Source => {
                scorer
                    .score_source_pairs_concurrent(&all_query_doc_pairs, &documents, concurrency)
                    .await?
            }
            ScoringMode::Pooled => {
                if is_semantic_only {
                    // Use embedding-based pooling for semantic queries
                    info!("Semantic queries detected: using embedding-based pooling (HNSW)");
                    let embedding_client = EmbeddingClient::new(&embedding_base_url, &model);
                    scorer
                        .score_pooled_semantic(&all_queries, &documents, &embedding_client, pool_size, concurrency)
                        .await?
                } else {
                    scorer
                        .score_pooled_with_options(&all_queries, &documents, pool_size, concurrency, false)
                        .await?
                }
            }
            ScoringMode::Exhaustive => {
                let qrels = scorer
                    .score_exhaustive(&all_queries, &documents, concurrency)
                    .await?;
                // For semantic mode, filter out qrels where query has word overlap with doc
                if is_semantic_only {
                    filter_semantic_qrels(qrels, &all_queries, &documents)
                } else {
                    qrels
                }
            }
        };

        // Write output based on format
        match dataset_info.format {
            DatasetFormat::Beir => {
                let queries_path = effective_dest.join("queries.jsonl");
                write_queries(&queries_path, &all_queries)?;
                info!("Wrote {} queries to {}", all_queries.len(), queries_path.display());

                if splits.len() > 1 {
                    let qrels_dir = effective_dest.join("qrels");
                    std::fs::create_dir_all(&qrels_dir)?;

                    let total_original: usize = splits.iter().map(|s| s.count).sum();
                    let mut offset = 0;

                    for split in &splits {
                        let ratio = split.count as f64 / total_original as f64;
                        let split_count = (all_qrels.len() as f64 * ratio).round() as usize;
                        let split_count = split_count.min(all_qrels.len() - offset);

                        let split_qrels: Vec<_> = all_qrels[offset..offset + split_count].to_vec();
                        let split_path = qrels_dir.join(format!("{}.tsv", split.name));
                        write_qrels(&split_path, &split_qrels)?;
                        info!(
                            "Wrote {} qrels to {} (preserving {:.1}% ratio)",
                            split_qrels.len(),
                            split_path.display(),
                            ratio * 100.0
                        );

                        offset += split_count;
                    }
                } else {
                    let qrels_path = effective_dest.join("qrels.tsv");
                    write_qrels(&qrels_path, &all_qrels)?;
                    info!("Wrote {} qrels to {}", all_qrels.len(), qrels_path.display());
                }
            }
            DatasetFormat::Ocr => {
                let queries_path = effective_dest.join("queries.json");
                write_ocr_queries(&queries_path, &all_queries, &all_qrels)?;
                info!(
                    "Wrote {} queries with {} relevance judgments to {}",
                    all_queries.len(),
                    all_qrels.len(),
                    queries_path.display()
                );
            }
        }

        info!("Completed {} for {}", locale_label, effective_dest.display());
    }

    if in_place {
        info!("Replaced queries and qrels in-place at {}", dest_dir.display());
    } else {
        info!("Created remixed dataset at {}", dest_dir.display());
    }
    Ok(())
}

/// Find qrels files in a specific directory (for locale-specific processing)
fn find_qrels_files_in_dir(path: &Path) -> Vec<PathBuf> {
    let mut qrels = Vec::new();

    // Check for qrels directory
    let qrels_dir = path.join("qrels");
    if qrels_dir.is_dir() {
        if let Ok(entries) = std::fs::read_dir(&qrels_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.extension().map_or(false, |e| e == "tsv") {
                    qrels.push(entry_path);
                }
            }
        }
    }

    // Also check for qrels.tsv directly in the directory
    let direct_qrels = path.join("qrels.tsv");
    if direct_qrels.exists() {
        qrels.push(direct_qrels);
    }

    qrels.sort();
    qrels
}

/// Recursively copy a directory
fn copy_dir_recursive(src: &Path, dst: &Path) -> Result<()> {
    std::fs::create_dir_all(dst)?;

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if src_path.is_dir() {
            copy_dir_recursive(&src_path, &dst_path)?;
        } else {
            std::fs::copy(&src_path, &dst_path)?;
        }
    }

    Ok(())
}

/// Batch remix all compatible datasets in a directory
async fn run_remix_batch(
    source_dir: PathBuf,
    output_mode: OutputMode,
    output_dir: Option<PathBuf>,
    on_exist: OnExist,
    query_types: Vec<QueryType>,
    queries_per_type: Option<usize>,
    base_url: String,
    model: String,
    api_key: String,
    concurrency: usize,
    scoring_mode: ScoringMode,
    pool_size: usize,
    dry_run: bool,
    trust_locale: bool,
) -> Result<()> {
    info!("Discovering datasets in {}...", source_dir.display());

    let discovery = discover_datasets(&source_dir)?;

    if discovery.datasets.is_empty() {
        info!("No compatible datasets found");
        if !discovery.skipped.is_empty() {
            info!("Skipped {} incompatible directories", discovery.skipped.len());
        }
        return Ok(());
    }

    info!(
        "Found {} compatible datasets, {} skipped",
        discovery.datasets.len(),
        discovery.skipped.len()
    );

    // Build suffix from query types
    let type_suffix = if query_types.len() == 1 {
        query_types[0].as_str().to_string()
    } else {
        query_types
            .iter()
            .map(|t| t.as_str())
            .collect::<Vec<_>>()
            .join("-")
    };

    // Determine output base directory for grouped mode
    let grouped_output_dir = output_dir.unwrap_or_else(|| {
        let name = source_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("datasets");
        source_dir
            .parent()
            .unwrap_or(&source_dir)
            .join(format!("{}-{}", name, type_suffix))
    });

    let mut succeeded = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let total = discovery.datasets.len();

    for (idx, dataset_info) in discovery.datasets.iter().enumerate() {
        let dataset_name = dataset_info
            .root_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        let output_name = format!("{}-{}", dataset_name, type_suffix);

        // Determine output path based on mode
        let output_path = match output_mode {
            OutputMode::Sibling => source_dir.join(&output_name),
            OutputMode::Grouped => grouped_output_dir.join(&output_name),
        };

        info!(
            "\n[{}/{}] {} → {}",
            idx + 1,
            total,
            dataset_name,
            output_name
        );

        // Check if output already exists
        if output_path.exists() {
            match on_exist {
                OnExist::Skip => {
                    info!("  Skipping: output already exists");
                    skipped += 1;
                    continue;
                }
                OnExist::Overwrite => {
                    info!("  Removing existing output...");
                    if !dry_run {
                        std::fs::remove_dir_all(&output_path)?;
                    }
                }
                OnExist::Ask => {
                    // For now, treat ask as skip in non-interactive mode
                    // TODO: Add interactive prompt support
                    info!("  Skipping: output already exists (use --on-exist overwrite to replace)");
                    skipped += 1;
                    continue;
                }
            }
        }

        if dry_run {
            info!("  [DRY RUN] Would create: {}", output_path.display());
            succeeded += 1;
            continue;
        }

        // Compute output directory (parent of the dataset)
        let parent_output = match output_mode {
            OutputMode::Sibling => Some(source_dir.clone()),
            OutputMode::Grouped => Some(grouped_output_dir.clone()),
        };

        // Run remix for this dataset
        match run_remix(
            dataset_info.root_dir.clone(),
            Some(output_name.clone()),
            parent_output,
            false, // not in-place
            query_types.clone(),
            queries_per_type,
            base_url.clone(),
            model.clone(),
            api_key.clone(),
            concurrency,
            scoring_mode,
            pool_size,
            dry_run,
            trust_locale,
        )
        .await
        {
            Ok(_) => {
                succeeded += 1;
            }
            Err(e) => {
                info!("  Failed: {}", e);
                failed += 1;
            }
        }
    }

    // Print summary
    info!("\n========================================");
    info!(
        "Batch remix complete: {} succeeded, {} failed, {} skipped",
        succeeded, failed, skipped
    );

    if failed > 0 {
        anyhow::bail!("{} dataset(s) failed to remix", failed);
    }

    Ok(())
}
