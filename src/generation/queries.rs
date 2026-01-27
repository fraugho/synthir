use crate::checkpoint::{CheckpointManager, ProgressState};
use crate::config::QueryType;
use crate::llm::{
    academic_query_prompt, basic_query_prompt, complex_query_prompt, keyword_query_prompt,
    natural_query_prompt, semantic_query_prompt, with_language_instruction, LLMProvider,
};
use crate::output::{append_query, BeirDocument, BeirQuery};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::{IndexedRandom, SliceRandom};
use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn};

/// Maximum retries for semantic queries with word overlap
const MAX_SEMANTIC_RETRIES: usize = 3;

/// Minimum word length to consider for overlap checking
const MIN_WORD_LENGTH: usize = 3;

/// Check if query has word overlap with document text
/// Returns the overlapping words if any
fn check_word_overlap(query: &str, document: &str) -> Vec<String> {
    // Normalize and tokenize document
    let doc_words: HashSet<String> = document
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= MIN_WORD_LENGTH)
        .map(|w| w.to_string())
        .collect();

    // Tokenize query and check against document
    let query_lower = query.to_lowercase();
    let query_words: Vec<&str> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= MIN_WORD_LENGTH)
        .collect();

    let mut overlaps = Vec::new();
    for word in query_words {
        // Check exact match
        if doc_words.contains(word) {
            overlaps.push(word.to_string());
            continue;
        }
        // Check stem overlap (simple: check if word is prefix/suffix of any doc word or vice versa)
        for doc_word in &doc_words {
            if word.len() >= 4 && doc_word.len() >= 4 {
                // Check if they share a common stem (first 4+ chars)
                let min_len = word.len().min(doc_word.len()).min(6);
                if word[..min_len] == doc_word[..min_len] {
                    overlaps.push(format!("{}~{}", word, doc_word));
                    break;
                }
            }
        }
    }

    overlaps
}

/// Generate queries for documents
pub struct QueryGenerator<'a> {
    provider: &'a LLMProvider,
    dry_run: bool,
    language: Option<String>,
}

impl<'a> QueryGenerator<'a> {
    pub fn new(provider: &'a LLMProvider, dry_run: bool) -> Self {
        Self { provider, dry_run, language: None }
    }

    /// Set the target language for query generation
    pub fn with_language(mut self, language: Option<String>) -> Self {
        self.language = language;
        self
    }

    /// Generate a single query for a document
    pub async fn generate_one(
        &self,
        doc: &BeirDocument,
        query_type: QueryType,
        index: usize,
    ) -> Result<BeirQuery> {
        let query_id = format!("{}_{:06}", query_type.as_str(), index + 1);

        if self.dry_run {
            return Ok(BeirQuery {
                id: query_id,
                text: format!(
                    "[DRY RUN] {} query for doc {}",
                    query_type.as_str(),
                    doc.id
                ),
            });
        }

        let base_prompt = match query_type {
            QueryType::Natural => natural_query_prompt(&doc.text),
            QueryType::Keyword => keyword_query_prompt(&doc.text),
            QueryType::Academic => academic_query_prompt(&doc.text),
            QueryType::Complex => complex_query_prompt(&doc.text),
            QueryType::Semantic => semantic_query_prompt(&doc.text),
            QueryType::Basic => basic_query_prompt(&doc.text),
            QueryType::Mixed => {
                // Should not be called directly for mixed type
                unreachable!("Mixed query type should be handled by generate_mixed")
            }
        };

        // Add language instruction if specified
        let prompt = with_language_instruction(base_prompt, self.language.as_deref());

        // For semantic queries, retry if there's word overlap
        if query_type == QueryType::Semantic {
            let text = self.generate_semantic_with_retry(&doc.text, &prompt).await?;
            return Ok(BeirQuery { id: query_id, text });
        }

        let text = self.provider.generate_query(&prompt).await?;

        Ok(BeirQuery { id: query_id, text })
    }

    /// Generate semantic query with retry on word overlap
    async fn generate_semantic_with_retry(
        &self,
        doc_text: &str,
        prompt: &str,
    ) -> Result<String> {
        let mut last_query = String::new();
        let mut last_overlaps = Vec::new();

        for attempt in 0..MAX_SEMANTIC_RETRIES {
            let query = self.provider.generate_query(prompt).await?;
            let overlaps = check_word_overlap(&query, doc_text);

            if overlaps.is_empty() {
                // No overlap, success!
                if attempt > 0 {
                    info!("Semantic query succeeded on attempt {}: '{}'", attempt + 1, query);
                }
                return Ok(query);
            }

            last_query = query.clone();
            last_overlaps = overlaps.clone();

            if attempt + 1 < MAX_SEMANTIC_RETRIES {
                warn!(
                    "Semantic query '{}' has word overlap {:?}, retrying ({}/{})",
                    query, overlaps, attempt + 1, MAX_SEMANTIC_RETRIES
                );
            }
        }

        // All retries exhausted, return last query with warning
        warn!(
            "Semantic query '{}' still has overlap {:?} after {} retries, using anyway",
            last_query, last_overlaps, MAX_SEMANTIC_RETRIES
        );
        Ok(last_query)
    }

    /// Generate queries for all documents of a specific type
    pub async fn generate_for_type(
        &self,
        documents: &[BeirDocument],
        query_type: QueryType,
        queries_per_type: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
    ) -> Result<Vec<(BeirQuery, String)>> {
        self.generate_for_type_concurrent(
            documents,
            query_type,
            queries_per_type,
            output_path,
            state,
            checkpoint_mgr,
            1,
        )
        .await
    }

    /// Generate queries for all documents with concurrent execution
    pub async fn generate_for_type_concurrent(
        &self,
        documents: &[BeirDocument],
        query_type: QueryType,
        queries_per_type: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
        concurrency: usize,
    ) -> Result<Vec<(BeirQuery, String)>> {
        // Vec of (query, source_doc_id)
        let mut results: Vec<(BeirQuery, String)> = Vec::new();
        let concurrency = concurrency.max(1);

        let pb = ProgressBar::new(queries_per_type as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(&format!(
                    "{{spinner:.green}} [{{elapsed_precise}}] [{{bar:40.cyan/blue}}] {{pos}}/{{len}} {} queries ({{eta}})",
                    query_type
                ))
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut rng = rand::rng();
        let mut doc_indices: Vec<usize> = (0..documents.len()).collect();

        // Collect pending queries
        let mut pending: Vec<(usize, usize)> = Vec::new(); // (query_index, doc_index)
        let mut generated = 0;
        while generated < queries_per_type {
            doc_indices.shuffle(&mut rng);

            for &doc_idx in &doc_indices {
                if generated >= queries_per_type {
                    break;
                }

                let query_id = format!("{}_{:06}", query_type.as_str(), generated + 1);

                if state.is_query_completed(query_type, &query_id) {
                    pb.inc(1);
                } else {
                    pending.push((generated, doc_idx));
                }
                generated += 1;
            }
        }

        if pending.is_empty() {
            pb.finish_with_message(format!("{} query generation complete (all cached)", query_type));
            info!("All {} {} queries already generated", queries_per_type, query_type);
            return Ok(results);
        }

        // Generate queries concurrently in batches
        let semaphore = Arc::new(Semaphore::new(concurrency));

        for chunk in pending.chunks(concurrency * 2) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|&(query_idx, doc_idx)| {
                    let sem = semaphore.clone();
                    let doc = documents[doc_idx].clone();
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = self.generate_one(&doc, query_type, query_idx).await;
                        (query_idx, doc.id.clone(), result)
                    }
                })
                .collect();

            let batch_results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            // Sort by query index to maintain order
            let mut sorted_results: Vec<_> = batch_results.into_iter().collect();
            sorted_results.sort_by_key(|(idx, _, _)| *idx);

            for (query_idx, doc_id, result) in sorted_results {
                let query = result?;
                let query_id = format!("{}_{:06}", query_type.as_str(), query_idx + 1);

                // Append to file
                append_query(output_path, &query)?;

                // Update progress
                state.mark_query_completed(query_type, &query_id);
                checkpoint_mgr.record_and_maybe_save(state)?;

                results.push((query, doc_id));
                pb.inc(1);
            }
        }

        pb.finish_with_message(format!("{} query generation complete", query_type));
        info!("Generated {} {} queries", results.len(), query_type);

        Ok(results)
    }

    /// Generate mixed queries (sample from all types)
    pub async fn generate_mixed(
        &self,
        documents: &[BeirDocument],
        total_queries: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
    ) -> Result<Vec<(BeirQuery, String, QueryType)>> {
        // Vec of (query, source_doc_id, query_type)
        let mut results: Vec<(BeirQuery, String, QueryType)> = Vec::new();

        let pb = ProgressBar::new(total_queries as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} mixed queries ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut rng = rand::rng();
        let query_types = QueryType::all_types();

        for i in 0..total_queries {
            let query_id = format!("mixed_{:06}", i + 1);

            // Skip if already completed
            if state.is_query_completed(QueryType::Mixed, &query_id) {
                pb.inc(1);
                continue;
            }

            // Pick random document and query type
            let doc = documents.choose(&mut rng).unwrap();
            let query_type = *query_types.choose(&mut rng).unwrap();

            let text = if self.dry_run {
                format!(
                    "[DRY RUN] {} query for doc {}",
                    query_type.as_str(),
                    doc.id
                )
            } else {
                let base_prompt = match query_type {
                    QueryType::Natural => natural_query_prompt(&doc.text),
                    QueryType::Keyword => keyword_query_prompt(&doc.text),
                    QueryType::Academic => academic_query_prompt(&doc.text),
                    QueryType::Complex => complex_query_prompt(&doc.text),
                    QueryType::Semantic => semantic_query_prompt(&doc.text),
                    QueryType::Basic => basic_query_prompt(&doc.text),
                    QueryType::Mixed => unreachable!(),
                };
                let prompt = with_language_instruction(base_prompt, self.language.as_deref());
                // Use retry logic for semantic queries in mixed mode too
                if query_type == QueryType::Semantic {
                    self.generate_semantic_with_retry(&doc.text, &prompt).await?
                } else {
                    self.provider.generate_query(&prompt).await?
                }
            };

            let query = BeirQuery { id: query_id.clone(), text };

            // Append to file
            append_query(output_path, &query)?;

            // Update progress
            state.mark_query_completed(QueryType::Mixed, &query_id);
            checkpoint_mgr.record_and_maybe_save(state)?;

            results.push((query, doc.id.clone(), query_type));
            pb.inc(1);
        }

        pb.finish_with_message("Mixed query generation complete");
        info!("Generated {} mixed queries", results.len());

        Ok(results)
    }
}
