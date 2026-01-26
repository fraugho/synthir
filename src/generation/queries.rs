use crate::checkpoint::{CheckpointManager, ProgressState};
use crate::config::QueryType;
use crate::llm::{
    academic_query_prompt, basic_query_prompt, complex_query_prompt, keyword_query_prompt,
    natural_query_prompt, semantic_query_prompt, LLMProvider,
};
use crate::output::{append_query, BeirDocument, BeirQuery};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use rand::prelude::{IndexedRandom, SliceRandom};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::info;

/// Generate queries for documents
pub struct QueryGenerator<'a> {
    provider: &'a LLMProvider,
    dry_run: bool,
}

impl<'a> QueryGenerator<'a> {
    pub fn new(provider: &'a LLMProvider, dry_run: bool) -> Self {
        Self { provider, dry_run }
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

        let prompt = match query_type {
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

        let text = self.provider.generate_query(&prompt).await?;

        Ok(BeirQuery { id: query_id, text })
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
                let prompt = match query_type {
                    QueryType::Natural => natural_query_prompt(&doc.text),
                    QueryType::Keyword => keyword_query_prompt(&doc.text),
                    QueryType::Academic => academic_query_prompt(&doc.text),
                    QueryType::Complex => complex_query_prompt(&doc.text),
                    QueryType::Semantic => semantic_query_prompt(&doc.text),
                    QueryType::Basic => basic_query_prompt(&doc.text),
                    QueryType::Mixed => unreachable!(),
                };
                self.provider.generate_query(&prompt).await?
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
