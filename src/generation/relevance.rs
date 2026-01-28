use crate::config::ScoreScale;
use crate::llm::{fine_grained_relevance_prompt, range_relevance_prompt, relevance_scoring_prompt, EmbeddingClient, LLMProvider};
use crate::output::{BeirDocument, BeirQuery, Qrel};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use hnsw::{Hnsw, Searcher};
use indicatif::{ProgressBar, ProgressStyle};
use space::Metric;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument};
use tokio::sync::Semaphore;
use tracing::{debug, info};

/// Minimum word length for overlap checking
const MIN_WORD_LENGTH: usize = 3;

/// Check if query has word overlap with document text
/// Returns true if there's any overlap
fn has_word_overlap(query: &str, document: &str) -> bool {
    // Normalize and tokenize document
    let doc_words: HashSet<String> = document
        .to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MIN_WORD_LENGTH)
        .map(|w| w.to_string())
        .collect();

    // Tokenize query
    let query_lower = query.to_lowercase();
    let query_words: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= MIN_WORD_LENGTH)
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

/// Score relevance between queries and documents
pub struct RelevanceScorer<'a> {
    provider: &'a LLMProvider,
    dry_run: bool,
    score_scale: ScoreScale,
    score_min: u16,
    score_max: u16,
}

impl<'a> RelevanceScorer<'a> {
    pub fn new(provider: &'a LLMProvider, dry_run: bool) -> Self {
        Self {
            provider,
            dry_run,
            score_scale: ScoreScale::Trec,
            score_min: 0,
            score_max: 3,
        }
    }

    /// Create a new scorer with custom score scale and range
    pub fn with_scale(provider: &'a LLMProvider, dry_run: bool, score_scale: ScoreScale, score_min: u16, score_max: u16) -> Self {
        Self {
            provider,
            dry_run,
            score_scale,
            score_min,
            score_max,
        }
    }

    /// Score a single query-document pair
    pub async fn score_one(&self, query: &BeirQuery, doc: &BeirDocument) -> Result<u16> {
        if self.dry_run {
            // Return a random-ish score for dry run
            let hash = query.id.len() + doc.id.len();
            let range = self.score_max - self.score_min + 1;
            return Ok(self.score_min + (hash as u16 % range));
        }

        match self.score_scale {
            ScoreScale::Trec => {
                let prompt = relevance_scoring_prompt(&query.text, &doc.text);
                let score = self.provider.score_relevance(&prompt).await?;
                Ok(score as u16)
            }
            ScoreScale::Range => {
                let prompt = range_relevance_prompt(&query.text, &doc.text, self.score_min, self.score_max);
                self.provider.score_range(&prompt, self.score_min, self.score_max).await
            }
        }
    }

    /// Score queries against their source documents (known relevant pairs)
    /// Returns qrels for the source document pairs
    pub async fn score_source_pairs(
        &self,
        query_doc_pairs: &[(BeirQuery, String)], // (query, source_doc_id)
        documents: &[BeirDocument],
    ) -> Result<Vec<Qrel>> {
        self.score_source_pairs_concurrent(query_doc_pairs, documents, 1)
            .await
    }

    /// Score queries against their source documents with concurrency
    pub async fn score_source_pairs_concurrent(
        &self,
        query_doc_pairs: &[(BeirQuery, String)], // (query, source_doc_id)
        documents: &[BeirDocument],
        concurrency: usize,
    ) -> Result<Vec<Qrel>> {
        let doc_map: HashMap<&str, &BeirDocument> =
            documents.iter().map(|d| (d.id.as_str(), d)).collect();
        let concurrency = concurrency.max(1);

        let pb = ProgressBar::new(query_doc_pairs.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} relevance scores ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Collect valid pairs with documents
        let valid_pairs: Vec<_> = query_doc_pairs
            .iter()
            .enumerate()
            .filter_map(|(idx, (query, source_doc_id))| {
                doc_map
                    .get(source_doc_id.as_str())
                    .map(|doc| (idx, query.clone(), (*doc).clone(), source_doc_id.clone()))
            })
            .collect();

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut qrels = Vec::new();

        // Process in batches
        for chunk in valid_pairs.chunks(concurrency * 2) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|(idx, query, doc, source_doc_id)| {
                    let sem = semaphore.clone();
                    let q = query.clone();
                    let d = doc.clone();
                    let doc_id = source_doc_id.clone();
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = self.score_one(&q, &d).await;
                        (*idx, q.id.clone(), doc_id, result)
                    }
                })
                .collect();

            let batch_results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            // Sort by index to maintain order
            let mut sorted_results: Vec<_> = batch_results.into_iter().collect();
            sorted_results.sort_by_key(|(idx, _, _, _)| *idx);

            for (_, query_id, doc_id, result) in sorted_results {
                let score = result?;
                qrels.push(Qrel {
                    query_id,
                    doc_id,
                    score,
                });
                pb.inc(1);
            }
        }

        pb.finish_with_message("Source pair scoring complete");
        info!("Scored {} query-document source pairs", qrels.len());

        Ok(qrels)
    }

    /// Score queries against all documents to find additional relevant ones
    /// This is expensive but creates richer relevance judgments
    pub async fn score_cross_document(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        existing_qrels: &[Qrel],
        sample_size: usize, // How many docs to sample per query
    ) -> Result<Vec<Qrel>> {
        // Build set of existing (query_id, doc_id) pairs to skip
        let existing: std::collections::HashSet<(String, String)> = existing_qrels
            .iter()
            .map(|qrel| (qrel.query_id.clone(), qrel.doc_id.clone()))
            .collect();

        let mut qrels = Vec::new();
        let mut rng = rand::rng();

        let total = queries.len() * sample_size.min(documents.len());
        let pb = ProgressBar::new(total as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} cross-doc scores ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for query in queries {
            // Sample random documents
            use rand::prelude::SliceRandom;
            let mut sampled_docs: Vec<&BeirDocument> = documents.iter().collect();
            sampled_docs.shuffle(&mut rng);
            sampled_docs.truncate(sample_size);

            for doc in sampled_docs {
                // Skip if already scored
                if existing.contains(&(query.id.clone(), doc.id.clone())) {
                    pb.inc(1);
                    continue;
                }

                let score = self.score_one(query, doc).await?;

                // Only record if there's some relevance
                if score > 0 {
                    qrels.push(Qrel {
                        query_id: query.id.clone(),
                        doc_id: doc.id.clone(),
                        score,
                    });
                }

                pb.inc(1);
            }
        }

        pb.finish_with_message("Cross-document scoring complete");
        info!(
            "Found {} additional relevant pairs from cross-document scoring",
            qrels.len()
        );

        Ok(qrels)
    }

    /// Score a single query-document pair with fine-grained scale (0-100)
    /// Used internally for pooled scoring cliff detection
    pub async fn score_one_fine(&self, query: &BeirQuery, doc: &BeirDocument) -> Result<u8> {
        if self.dry_run {
            // Return a random-ish score for dry run
            let hash = (query.id.len() * 7 + doc.id.len() * 13) % 101;
            return Ok(hash as u8);
        }

        let prompt = fine_grained_relevance_prompt(&query.text, &doc.text);
        self.provider.score_fine_grained(&prompt).await
    }

    /// Score a single query-document pair with range scale
    pub async fn score_one_range(&self, query: &BeirQuery, doc: &BeirDocument) -> Result<u16> {
        if self.dry_run {
            let hash = (query.id.len() * 7 + doc.id.len() * 13) as u16;
            let range = self.score_max - self.score_min + 1;
            return Ok(self.score_min + (hash % range));
        }

        let prompt = range_relevance_prompt(&query.text, &doc.text, self.score_min, self.score_max);
        self.provider.score_range(&prompt, self.score_min, self.score_max).await
    }

    /// Build a BM25 index from documents
    fn build_bm25_index(&self, documents: &[BeirDocument]) -> Result<(Index, Schema)> {
        let mut schema_builder = Schema::builder();
        schema_builder.add_text_field("doc_id", STRING | STORED);
        schema_builder.add_text_field("text", TEXT);
        let schema = schema_builder.build();

        let index = Index::create_in_ram(schema.clone());
        let mut index_writer: IndexWriter = index.writer(50_000_000)?;

        let doc_id_field = schema.get_field("doc_id").unwrap();
        let text_field = schema.get_field("text").unwrap();

        for document in documents {
            index_writer.add_document(doc!(
                doc_id_field => document.id.clone(),
                text_field => format!("{} {}", document.title, document.text),
            ))?;
        }

        index_writer.commit()?;

        Ok((index, schema))
    }

    /// Get top-k candidate documents for a query using BM25
    fn get_bm25_candidates(
        &self,
        query: &BeirQuery,
        index: &Index,
        schema: &Schema,
        top_k: usize,
    ) -> Result<Vec<String>> {
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();

        let text_field = schema.get_field("text").unwrap();
        let doc_id_field = schema.get_field("doc_id").unwrap();

        let query_parser = QueryParser::for_index(index, vec![text_field]);

        // Escape special characters
        let escaped_query = escape_query(&query.text);
        let parsed_query = match query_parser.parse_query(&escaped_query) {
            Ok(q) => q,
            Err(_) => {
                // Fallback to simple term query if parsing fails
                let terms: Vec<&str> = query.text.split_whitespace().take(3).collect();
                query_parser.parse_query(&terms.join(" "))?
            }
        };

        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k))?;

        let mut candidates = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(doc_id) = retrieved_doc.get_first(doc_id_field) {
                if let Some(doc_id_str) = doc_id.as_str() {
                    candidates.push(doc_id_str.to_string());
                }
            }
        }

        Ok(candidates)
    }

    /// Detect cliff in sorted scores - returns index where cliff occurs
    /// A cliff is the largest gap between consecutive scores
    fn detect_cliff(scores: &[(String, u8)]) -> usize {
        if scores.len() <= 1 {
            return scores.len();
        }

        let mut max_gap = 0u8;
        let mut cliff_idx = scores.len();

        for i in 0..scores.len() - 1 {
            let gap = scores[i].1.saturating_sub(scores[i + 1].1);
            // Only consider it a cliff if gap is significant (>15 points) and score is above threshold
            if gap > max_gap && gap >= 15 && scores[i].1 >= 40 {
                max_gap = gap;
                cliff_idx = i + 1;
            }
        }

        // If no significant cliff found, use a threshold approach
        if cliff_idx == scores.len() {
            // Take all docs with score >= 50
            for (i, (_, score)) in scores.iter().enumerate() {
                if *score < 50 {
                    return i;
                }
            }
        }

        cliff_idx
    }

    /// Convert fine-grained score (0-100) to TREC score (0-3)
    fn fine_to_trec(score: u8) -> u16 {
        match score {
            0..=25 => 0,
            26..=50 => 1,
            51..=75 => 2,
            76..=100 => 3,
            _ => 3, // Shouldn't happen, but treat as highly relevant
        }
    }

    /// Convert fine-grained score (0-100) to custom range
    fn fine_to_range(score: u8, min: u16, max: u16) -> u16 {
        // Linear mapping from 0-100 to min-max
        let range = max - min;
        min + ((score as u16 * range) / 100)
    }

    /// Convert fine-grained score to the configured output scale
    fn fine_to_output(&self, score: u8) -> u16 {
        match self.score_scale {
            ScoreScale::Trec => Self::fine_to_trec(score),
            ScoreScale::Range => Self::fine_to_range(score, self.score_min, self.score_max),
        }
    }

    /// Score queries using BM25 pooling + cliff detection for many-to-many mappings
    /// When semantic_mode is true, filters out candidates that have word overlap with the query
    pub async fn score_pooled(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        pool_size: usize,
        concurrency: usize,
    ) -> Result<Vec<Qrel>> {
        self.score_pooled_with_options(queries, documents, pool_size, concurrency, false).await
    }

    /// Score queries using BM25 pooling with semantic filtering option
    /// When semantic_mode is true, filters out candidates that have word overlap with the query
    pub async fn score_pooled_with_options(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        pool_size: usize,
        concurrency: usize,
        semantic_mode: bool,
    ) -> Result<Vec<Qrel>> {
        info!("Building BM25 index for {} documents...", documents.len());
        let (index, schema) = self.build_bm25_index(documents)?;

        let doc_map: HashMap<&str, &BeirDocument> =
            documents.iter().map(|d| (d.id.as_str(), d)).collect();

        let concurrency = concurrency.max(1);
        let mut all_qrels = Vec::new();
        let mut total_filtered = 0usize;

        let pb = ProgressBar::new(queries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} queries pooled ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for query in queries {
            // Step 1: Get BM25 candidates (get more than needed since we'll filter)
            let fetch_size = if semantic_mode { pool_size * 3 } else { pool_size };
            let mut candidates = self.get_bm25_candidates(query, &index, &schema, fetch_size)?;

            // Step 1.5: For semantic mode, filter out candidates with word overlap
            if semantic_mode {
                let before_count = candidates.len();
                candidates.retain(|doc_id| {
                    if let Some(doc) = doc_map.get(doc_id.as_str()) {
                        let full_text = format!("{} {}", doc.title, doc.text);
                        !has_word_overlap(&query.text, &full_text)
                    } else {
                        false
                    }
                });
                let filtered = before_count - candidates.len();
                total_filtered += filtered;
                // Truncate to pool_size after filtering
                candidates.truncate(pool_size);
            }

            if candidates.is_empty() {
                pb.inc(1);
                continue;
            }

            // Step 2: Score each candidate with fine-grained scale (0-100)
            let semaphore = Arc::new(Semaphore::new(concurrency));
            let futures: Vec<_> = candidates
                .iter()
                .filter_map(|doc_id| doc_map.get(doc_id.as_str()).map(|d| (doc_id.clone(), *d)))
                .map(|(doc_id, doc)| {
                    let sem = semaphore.clone();
                    let q = query.clone();
                    let d = doc.clone();
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let score = self.score_one_fine(&q, &d).await;
                        (doc_id, score)
                    }
                })
                .collect();

            let results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            // Collect and sort scores
            let mut scored: Vec<(String, u8)> = Vec::new();
            for (doc_id, result) in results {
                if let Ok(score) = result {
                    scored.push((doc_id, score));
                }
            }
            scored.sort_by(|a, b| b.1.cmp(&a.1)); // Sort descending by score

            // Step 3: Detect cliff
            let cliff_idx = Self::detect_cliff(&scored);

            debug!(
                "Query {}: {} candidates, cliff at {}, scores: {:?}",
                query.id,
                scored.len(),
                cliff_idx,
                scored.iter().map(|(_, s)| *s).collect::<Vec<_>>()
            );

            // Step 4: Convert above-cliff docs to output scores, below-cliff as hard negatives (min)
            // No artificial capping - cliff detection determines relevant vs hard negative
            for (i, (doc_id, fine_score)) in scored.iter().enumerate() {
                let output_score = if i < cliff_idx {
                    self.fine_to_output(*fine_score)
                } else {
                    self.score_min // Hard negative
                };

                all_qrels.push(Qrel {
                    query_id: query.id.clone(),
                    doc_id: doc_id.clone(),
                    score: output_score,
                });
            }

            pb.inc(1);
        }

        pb.finish_with_message("Pooled scoring complete");
        if semantic_mode && total_filtered > 0 {
            info!(
                "Filtered {} candidate docs with word overlap (semantic mode)",
                total_filtered
            );
        }
        info!(
            "Generated {} relevance judgments using pooled scoring",
            all_qrels.len()
        );

        Ok(all_qrels)
    }

    /// Score every query against every document (exhaustive mode)
    /// This is expensive: O(queries * documents) LLM calls
    pub async fn score_exhaustive(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        concurrency: usize,
    ) -> Result<Vec<Qrel>> {
        let total_pairs = queries.len() * documents.len();
        info!(
            "Exhaustive scoring: {} queries x {} documents = {} pairs",
            queries.len(),
            documents.len(),
            total_pairs
        );

        let concurrency = concurrency.max(1);

        let pb = ProgressBar::new(total_pairs as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} exhaustive scores ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Build all query-document pairs
        let pairs: Vec<_> = queries
            .iter()
            .flat_map(|q| documents.iter().map(move |d| (q.clone(), d.clone())))
            .collect();

        let semaphore = Arc::new(Semaphore::new(concurrency));
        let mut all_qrels = Vec::new();

        // Process in chunks for memory efficiency
        for chunk in pairs.chunks(concurrency * 4) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|(query, doc)| {
                    let sem = semaphore.clone();
                    let q = query.clone();
                    let d = doc.clone();
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = self.score_one_fine(&q, &d).await;
                        (q.id.clone(), d.id.clone(), result)
                    }
                })
                .collect();

            let batch_results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            for (query_id, doc_id, result) in batch_results {
                if let Ok(fine_score) = result {
                    let output_score = self.fine_to_output(fine_score);
                    all_qrels.push(Qrel {
                        query_id,
                        doc_id,
                        score: output_score,
                    });
                }
                pb.inc(1);
            }
        }

        pb.finish_with_message("Exhaustive scoring complete");
        info!(
            "Generated {} relevance judgments using exhaustive scoring",
            all_qrels.len()
        );

        Ok(all_qrels)
    }

    /// Score queries using embedding-based pooling (for semantic queries)
    /// 1. Embed all documents and build HNSW index
    /// 2. For each query, embed and find top-K nearest neighbors
    /// 3. LLM scores those K candidates
    pub async fn score_pooled_semantic(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        embedding_client: &EmbeddingClient,
        pool_size: usize,
        concurrency: usize,
    ) -> Result<Vec<Qrel>> {
        let concurrency = concurrency.max(1);

        // Step 1: Embed all documents
        info!("Embedding {} documents for vector index...", documents.len());
        let pb_embed = ProgressBar::new(documents.len() as u64);
        pb_embed.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} docs embedded ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Batch embed documents
        let mut doc_embeddings: Vec<(String, Vec<f32>)> = Vec::new();
        let batch_size = 32; // Embed in batches

        for chunk in documents.chunks(batch_size) {
            let texts: Vec<String> = chunk
                .iter()
                .map(|d| format!("{} {}", d.title, d.text))
                .collect();

            let embeddings = embedding_client.embed_batch(&texts).await?;

            for (doc, emb) in chunk.iter().zip(embeddings.into_iter()) {
                doc_embeddings.push((doc.id.clone(), emb));
                pb_embed.inc(1);
            }
        }
        pb_embed.finish_with_message("Document embedding complete");

        if doc_embeddings.is_empty() {
            return Ok(vec![]);
        }

        // Get embedding dimension from first embedding
        let embed_dim = doc_embeddings[0].1.len();
        info!("Embedding dimension: {}", embed_dim);

        // Step 2: Build HNSW index
        info!("Building HNSW vector index...");
        // M=12, M0=24 are typical HNSW parameters
        // Use rand_pcg which is compatible with the rand_core version hnsw expects
        let mut hnsw: Hnsw<CosineMetric, Vec<f32>, rand_pcg::Pcg64, 12, 24> =
            Hnsw::new(CosineMetric);
        let mut doc_id_map: Vec<String> = Vec::new();

        for (doc_id, embedding) in &doc_embeddings {
            hnsw.insert(embedding.clone(), &mut Searcher::default());
            doc_id_map.push(doc_id.clone());
        }

        info!("HNSW index built with {} documents", doc_id_map.len());

        // Build doc map for scoring
        let doc_map: HashMap<String, BeirDocument> =
            documents.iter().map(|d| (d.id.clone(), d.clone())).collect();

        // Step 3: Batch embed all queries upfront
        info!("Embedding {} queries...", queries.len());
        let pb_query_embed = ProgressBar::new(queries.len() as u64);
        pb_query_embed.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} queries embedded ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let mut query_embeddings: HashMap<String, Vec<f32>> = HashMap::new();
        for chunk in queries.chunks(batch_size) {
            let texts: Vec<String> = chunk.iter().map(|q| q.text.clone()).collect();
            let embeddings = embedding_client.embed_batch(&texts).await?;

            for (query, emb) in chunk.iter().zip(embeddings.into_iter()) {
                query_embeddings.insert(query.id.clone(), emb);
                pb_query_embed.inc(1);
            }
        }
        pb_query_embed.finish_with_message("Query embedding complete");

        // Step 4: Process all queries concurrently with global semaphore
        let pb = ProgressBar::new(queries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} queries (semantic pooled) ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Global semaphore for LLM calls across all queries
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let hnsw = Arc::new(hnsw);
        let doc_id_map = Arc::new(doc_id_map);
        let doc_map = Arc::new(doc_map);
        let query_embeddings = Arc::new(query_embeddings);
        let pb = Arc::new(pb);

        let query_futures: Vec<_> = queries
            .iter()
            .map(|query| {
                let sem = semaphore.clone();
                let hnsw = hnsw.clone();
                let doc_id_map = doc_id_map.clone();
                let doc_map = doc_map.clone();
                let query_embeddings = query_embeddings.clone();
                let pb = pb.clone();
                let query = query.clone();
                let score_min = self.score_min;

                async move {
                    let mut qrels = Vec::new();

                    // Get pre-computed query embedding
                    let query_embedding = match query_embeddings.get(&query.id) {
                        Some(emb) => emb.clone(),
                        None => {
                            pb.inc(1);
                            return qrels;
                        }
                    };

                    // Vector search for top-K candidates (fast, in-memory)
                    let mut searcher = Searcher::default();
                    let mut neighbors = vec![space::Neighbor { index: 0, distance: 0 }; pool_size * 2];
                    let found_count = hnsw.nearest(&query_embedding, pool_size * 2, &mut searcher, &mut neighbors).len();
                    neighbors.truncate(found_count);

                    let mut candidates: Vec<String> = neighbors
                        .iter()
                        .map(|n| doc_id_map[n.index].clone())
                        .collect();

                    // Filter out candidates with word overlap
                    candidates.retain(|doc_id| {
                        if let Some(doc) = doc_map.get(doc_id) {
                            let full_text = format!("{} {}", doc.title, doc.text);
                            !has_word_overlap(&query.text, &full_text)
                        } else {
                            false
                        }
                    });
                    candidates.truncate(pool_size);

                    if candidates.is_empty() {
                        pb.inc(1);
                        return qrels;
                    }

                    // Score candidates with LLM (using global semaphore)
                    let candidate_futures: Vec<_> = candidates
                        .iter()
                        .filter_map(|doc_id| doc_map.get(doc_id).map(|d| (doc_id.clone(), d.clone())))
                        .map(|(doc_id, doc)| {
                            let sem = sem.clone();
                            let q = query.clone();
                            async move {
                                let _permit = sem.acquire().await.unwrap();
                                let score = self.score_one_fine(&q, &doc).await;
                                (doc_id, score)
                            }
                        })
                        .collect();

                    let results: Vec<_> = stream::iter(candidate_futures)
                        .buffer_unordered(pool_size)
                        .collect()
                        .await;

                    // Collect and sort scores
                    let mut scored: Vec<(String, u8)> = Vec::new();
                    for (doc_id, result) in results {
                        if let Ok(score) = result {
                            scored.push((doc_id, score));
                        }
                    }
                    scored.sort_by(|a, b| b.1.cmp(&a.1));

                    // Detect cliff and convert to output scores
                    let cliff_idx = Self::detect_cliff(&scored);

                    for (i, (doc_id, fine_score)) in scored.iter().enumerate() {
                        let output_score = if i < cliff_idx {
                            self.fine_to_output(*fine_score)
                        } else {
                            score_min
                        };

                        qrels.push(Qrel {
                            query_id: query.id.clone(),
                            doc_id: doc_id.clone(),
                            score: output_score,
                        });
                    }

                    pb.inc(1);
                    qrels
                }
            })
            .collect();

        // Run all queries concurrently - semaphore limits total LLM calls
        let all_results: Vec<Vec<Qrel>> = stream::iter(query_futures)
            .buffer_unordered(queries.len())
            .collect()
            .await;

        let all_qrels: Vec<Qrel> = all_results.into_iter().flatten().collect();

        pb.finish_with_message("Semantic pooled scoring complete");
        info!(
            "Generated {} relevance judgments using semantic pooled scoring",
            all_qrels.len()
        );

        Ok(all_qrels)
    }
}

/// Cosine distance metric for embeddings
struct CosineMetric;

impl Metric<Vec<f32>> for CosineMetric {
    type Unit = u32;

    fn distance(&self, a: &Vec<f32>, b: &Vec<f32>) -> Self::Unit {
        // Cosine distance = 1 - cosine_similarity
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        let cosine_sim = if norm_a > 0.0 && norm_b > 0.0 {
            dot / (norm_a * norm_b)
        } else {
            0.0
        };

        // Convert to distance: 0 = identical, u32::MAX = opposite
        // cosine_sim ranges from -1 to 1, so (1 - cosine_sim) ranges from 0 to 2
        let distance = (1.0 - cosine_sim) * (u32::MAX as f32 / 2.0);
        distance as u32
    }
}

/// Escape special characters for tantivy query parser
fn escape_query(query: &str) -> String {
    let special_chars = [
        '+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\',
        '/',
    ];
    let mut escaped = String::with_capacity(query.len() * 2);

    for c in query.chars() {
        if special_chars.contains(&c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }

    escaped
}
