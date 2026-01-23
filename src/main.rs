use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use synthir::{
    benchmark::{run_benchmark, BenchmarkConfig},
    checkpoint::{CheckpointManager, GenerationPhase, ProgressState},
    config::{parse_query_types, GenerationConfig, QueryType, RuntimeOptions, ScoreScale, ScoringMode},
    generation::{DocumentGenerator, QueryGenerator, RelevanceScorer},
    llm::{topic_generation_prompt, LLMProvider, LLMProviderConfig, MultiEndpointProvider},
    meta::{run_meta_generation, MetaConfig},
    mining::HardNegativeMiner,
    output::{
        read_corpus, write_qrels, write_queries, BeirQuery, DatasetMetadata, OutputOrganizer,
        QueryCounts,
    },
    topics::{create_custom_topic, get_builtin_topics, get_topic, TopicConfig},
    utils::{display_dry_run_info, display_topics, display_verbose_start},
};
use tracing::{info, Level};
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

        let qrels = match config.scoring_mode {
            ScoringMode::Source => {
                scorer
                    .score_source_pairs_concurrent(&all_query_doc_pairs, &documents, config.concurrency)
                    .await?
            }
            ScoringMode::Pooled => {
                // Collect all queries for pooled scoring
                let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
                scorer
                    .score_pooled(&all_queries, &documents, config.pool_size, config.concurrency)
                    .await?
            }
            ScoringMode::Exhaustive => {
                // Score every query against every document
                let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
                scorer
                    .score_exhaustive(&all_queries, &documents, config.concurrency)
                    .await?
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
