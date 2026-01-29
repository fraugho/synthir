//! Meta mode for generating multiple datasets automatically

use crate::checkpoint::{CheckpointManager, ProgressState};
use crate::config::{DocDiversity, GenerationConfig, QueryDiversity, QueryType, ScoreScale, ScoringMode};
use crate::generation::{DocumentGenerator, QueryGenerator, RelevanceScorer};
use crate::llm::{LLMProvider, LLMProviderConfig};
use crate::mining::HardNegativeMiner;
use crate::output::{
    write_qrels, write_queries, BeirDocument, DatasetMetadata, OutputOrganizer, QueryCounts,
};
use crate::topics::create_custom_topic;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tracing::info;

/// Configuration for meta generation
#[derive(Debug, Clone)]
pub struct MetaConfig {
    pub topic_count: usize,
    pub document_count: usize,
    pub queries_per_type: usize,
    pub output_dir: PathBuf,
    pub base_url: String,
    pub model: String,
    pub api_key: String,
    pub shared_corpus: bool,
    pub generate_hard_negatives: bool,
    pub concurrency: usize,
    pub batch_size: usize,
}

/// Metadata for the meta generation run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaMetadata {
    pub topics: Vec<String>,
    pub document_count_per_topic: usize,
    pub queries_per_type: usize,
    pub model_used: String,
    pub shared_corpus: bool,
}

/// Prompt for generating multiple topic names
fn multi_topic_generation_prompt(count: usize) -> String {
    format!(
        r#"Generate {} unique, specific topics for document collections.
NOT generic topics like "technology" or "science".
Be specific and diverse, like:
- "vintage motorcycle restoration guides"
- "sourdough bread troubleshooting"
- "apartment lease agreements"
- "D&D campaign session notes"
- "houseplant care tips"
- "budget travel itineraries"

Output ONLY the topic names, one per line, nothing else.
Generate exactly {} topics."#,
        count, count
    )
}

/// Generate multiple topic names using LLM
async fn generate_topics(provider: &LLMProvider, count: usize) -> Result<Vec<String>> {
    let prompt = multi_topic_generation_prompt(count);
    let response = provider.complete(None, &prompt).await?;

    let topics: Vec<String> = response
        .lines()
        .map(|line| line.trim().trim_start_matches('-').trim().to_string())
        .filter(|line| !line.is_empty())
        .take(count)
        .collect();

    if topics.len() < count {
        info!(
            "Warning: LLM generated {} topics instead of requested {}",
            topics.len(),
            count
        );
    }

    Ok(topics)
}

/// Run meta generation - generate multiple datasets
pub async fn run_meta_generation(config: MetaConfig) -> Result<()> {
    let provider = LLMProvider::new(LLMProviderConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model.clone(),
        max_retries: 3,
    })?;

    // Create output directory
    fs::create_dir_all(&config.output_dir)?;

    // Generate topics
    info!("Generating {} topics...", config.topic_count);
    let topics = generate_topics(&provider, config.topic_count).await?;

    info!("Generated topics:");
    for (i, topic) in topics.iter().enumerate() {
        info!("  {}. {}", i + 1, topic);
    }

    // Generate shared corpus if requested
    let shared_corpus: Option<Vec<BeirDocument>> = if config.shared_corpus {
        info!(
            "Generating shared corpus with {} documents...",
            config.document_count
        );

        let shared_dir = config.output_dir.join("shared-corpus");
        fs::create_dir_all(&shared_dir)?;

        // Use first topic for shared corpus style
        let topic_config = create_custom_topic(
            "miscellaneous".to_string(),
            "Mixed topic shared corpus".to_string(),
        );

        let gen_config = GenerationConfig {
            topic: "miscellaneous".to_string(),
            corpus_path: None,
            document_count: config.document_count,
            queries_per_type: config.queries_per_type,
            query_types: None,
            output_dir: shared_dir.clone(),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            generate_hard_negatives: config.generate_hard_negatives,
            batch_size: config.batch_size,
            concurrency: config.concurrency,
            scoring_mode: ScoringMode::Source,
            pool_size: 30,
            score_scale: ScoreScale::Trec,
            score_min: 0,
            score_max: 3,
            doc_diversity: DocDiversity::default(),
            query_diversity: QueryDiversity::default(),
            diversity_threshold: 0.85,
            doc_categories: None,
            embedding_url: None,
            embedding_model: None,
        };

        let mut state = ProgressState::new(gen_config.clone());
        let mut checkpoint_mgr = CheckpointManager::new(shared_dir.clone(), config.batch_size);

        let doc_gen = DocumentGenerator::new(&provider, topic_config, false);
        let corpus_path = shared_dir.join("corpus.jsonl");
        let docs = doc_gen
            .generate_all_concurrent(
                config.document_count,
                &corpus_path,
                &mut state,
                &mut checkpoint_mgr,
                config.concurrency,
            )
            .await?;

        Some(docs)
    } else {
        None
    };

    // Generate dataset for each topic
    for (idx, topic_name) in topics.iter().enumerate() {
        info!(
            "\n=== Generating dataset {}/{}: {} ===",
            idx + 1,
            topics.len(),
            topic_name
        );

        let sanitized_name = sanitize_topic_name(topic_name);
        let topic_dir = config.output_dir.join(&sanitized_name);
        fs::create_dir_all(&topic_dir)?;

        let topic_config = create_custom_topic(
            topic_name.clone(),
            format!("LLM-generated topic: {}", topic_name),
        );

        let gen_config = GenerationConfig {
            topic: topic_name.clone(),
            corpus_path: None,
            document_count: config.document_count,
            queries_per_type: config.queries_per_type,
            query_types: None,
            output_dir: topic_dir.clone(),
            base_url: config.base_url.clone(),
            model: config.model.clone(),
            generate_hard_negatives: config.generate_hard_negatives,
            batch_size: config.batch_size,
            concurrency: config.concurrency,
            scoring_mode: ScoringMode::Source,
            pool_size: 30,
            score_scale: ScoreScale::Trec,
            score_min: 0,
            score_max: 3,
            doc_diversity: DocDiversity::default(),
            query_diversity: QueryDiversity::default(),
            diversity_threshold: 0.85,
            doc_categories: None,
            embedding_url: None,
            embedding_model: None,
        };

        let organizer = OutputOrganizer::new(topic_dir.clone(), sanitized_name.clone());
        organizer.create_structure()?;

        let mut state = ProgressState::new(gen_config.clone());
        let mut checkpoint_mgr = CheckpointManager::new(topic_dir.clone(), config.batch_size);

        // Phase 1: Documents
        let documents = if let Some(ref corpus) = shared_corpus {
            info!("Using shared corpus...");
            // Copy shared corpus to topic directory
            let corpus_path = organizer.corpus_path();
            for doc in corpus {
                crate::output::append_document(&corpus_path, doc)?;
            }
            corpus.clone()
        } else {
            info!(
                "Generating {} documents (concurrency: {})...",
                config.document_count, config.concurrency
            );
            let doc_gen = DocumentGenerator::new(&provider, topic_config.clone(), false);
            doc_gen
                .generate_all_concurrent(
                    config.document_count,
                    &organizer.corpus_path(),
                    &mut state,
                    &mut checkpoint_mgr,
                    config.concurrency,
                )
                .await?
        };

        // Phase 2: Queries
        info!(
            "Generating queries (concurrency: {})...",
            config.concurrency
        );
        let query_gen = QueryGenerator::new(&provider, false);

        let mut all_query_doc_pairs = Vec::new();

        for query_type in QueryType::all_types() {
            let pairs = query_gen
                .generate_for_type_concurrent(
                    &documents,
                    query_type,
                    config.queries_per_type,
                    &organizer.queries_path(query_type),
                    &mut state,
                    &mut checkpoint_mgr,
                    config.concurrency,
                )
                .await?;
            all_query_doc_pairs.extend(pairs);
        }

        // Mixed queries
        let _mixed_pairs = query_gen
            .generate_mixed(
                &documents,
                config.queries_per_type,
                &organizer.queries_path(QueryType::Mixed),
                &mut state,
                &mut checkpoint_mgr,
            )
            .await?;

        // Phase 3: Relevance scoring
        info!(
            "Scoring relevance (concurrency: {})...",
            config.concurrency
        );
        let scorer = RelevanceScorer::new(&provider, false);
        let qrels = scorer
            .score_source_pairs_concurrent(&all_query_doc_pairs, &documents, config.concurrency)
            .await?;

        // Write qrels for each query type
        for query_type in QueryType::all_types() {
            let type_qrels: Vec<_> = qrels
                .iter()
                .filter(|q| q.query_id.starts_with(query_type.as_str()))
                .cloned()
                .collect();
            write_qrels(&organizer.qrels_path(query_type), &type_qrels)?;
        }

        // Phase 4: Hard negatives (optional)
        if config.generate_hard_negatives {
            info!("Mining hard negatives...");
            let miner = HardNegativeMiner::new(&provider, false);

            let all_queries: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();

            let hard_negatives = miner
                .mine_hard_negatives(&all_queries, &documents, &qrels, 3)
                .await?;

            let mut merged_qrels = qrels.clone();
            merged_qrels.extend(hard_negatives);

            write_qrels(
                &organizer.hard_negatives_merged_dir().join("qrels.tsv"),
                &merged_qrels,
            )?;

            let all_query_refs: Vec<_> =
                all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
            write_queries(
                &organizer.hard_negatives_merged_dir().join("queries.jsonl"),
                &all_query_refs,
            )?;
        }

        // Write merged outputs
        let all_query_refs: Vec<_> = all_query_doc_pairs.iter().map(|(q, _)| q.clone()).collect();
        write_queries(
            &organizer.general_merged_dir().join("queries.jsonl"),
            &all_query_refs,
        )?;
        write_qrels(&organizer.general_merged_dir().join("qrels.tsv"), &qrels)?;

        // Write dataset metadata
        let metadata = DatasetMetadata {
            topic: topic_name.clone(),
            document_count: documents.len(),
            query_counts: QueryCounts {
                natural: config.queries_per_type,
                keyword: config.queries_per_type,
                academic: config.queries_per_type,
                complex: config.queries_per_type,
                semantic: config.queries_per_type,
                basic: config.queries_per_type,
                mixed: config.queries_per_type,
            },
            generation_timestamp: chrono_lite_timestamp(),
            model_used: config.model.clone(),
            has_hard_negatives: config.generate_hard_negatives,
        };
        organizer.write_metadata(&metadata)?;

        info!("Completed dataset: {}", topic_name);
    }

    // Write meta metadata
    let meta_metadata = MetaMetadata {
        topics: topics.clone(),
        document_count_per_topic: config.document_count,
        queries_per_type: config.queries_per_type,
        model_used: config.model.clone(),
        shared_corpus: config.shared_corpus,
    };

    let meta_path = config.output_dir.join("meta.json");
    let meta_content = serde_json::to_string_pretty(&meta_metadata)?;
    fs::write(meta_path, meta_content)?;

    info!("\n=== Meta generation complete! ===");
    info!("Generated {} datasets", topics.len());
    info!("Output directory: {}", config.output_dir.display());

    Ok(())
}

/// Sanitize topic name for use as directory name
fn sanitize_topic_name(name: &str) -> String {
    name.to_lowercase()
        .chars()
        .map(|c| if c.is_alphanumeric() || c == ' ' { c } else { '-' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join("-")
}

/// Simple timestamp
fn chrono_lite_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}", duration.as_secs())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_topic_name() {
        assert_eq!(
            sanitize_topic_name("Vintage Motorcycle Restoration"),
            "vintage-motorcycle-restoration"
        );
        assert_eq!(
            sanitize_topic_name("D&D Campaign Notes"),
            "d-d-campaign-notes"
        );
        assert_eq!(sanitize_topic_name("recipes"), "recipes");
    }

    #[test]
    fn test_multi_topic_prompt() {
        let prompt = multi_topic_generation_prompt(5);
        assert!(prompt.contains("5"));
        assert!(prompt.contains("unique"));
    }
}
