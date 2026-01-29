use crate::checkpoint::{CheckpointManager, ProgressState};
use crate::config::DocDiversity;
use crate::generation::diversity::{EmbeddingDiversityChecker, IdeaTracker};
use crate::llm::{
    document_expand_prompt, document_idea_prompt, document_system_prompt, document_user_prompt,
    EmbeddingClient, LLMProvider,
};
use crate::output::{append_document, BeirDocument};
use crate::topics::TopicConfig;
use anyhow::Result;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::{info, warn};

/// Configuration for document diversity
#[derive(Clone)]
pub struct DiversityConfig {
    pub mode: DocDiversity,
    pub threshold: f32,
    pub categories: Option<Vec<String>>,
    pub embedding_url: Option<String>,
    pub embedding_model: Option<String>,
}

impl Default for DiversityConfig {
    fn default() -> Self {
        Self {
            mode: DocDiversity::None,
            threshold: 0.85,
            categories: None,
            embedding_url: None,
            embedding_model: None,
        }
    }
}

/// Generate documents for a topic
pub struct DocumentGenerator<'a> {
    provider: &'a LLMProvider,
    topic_config: TopicConfig,
    dry_run: bool,
    diversity_config: DiversityConfig,
}

impl<'a> DocumentGenerator<'a> {
    pub fn new(provider: &'a LLMProvider, topic_config: TopicConfig, dry_run: bool) -> Self {
        Self {
            provider,
            topic_config,
            dry_run,
            diversity_config: DiversityConfig::default(),
        }
    }

    /// Configure diversity settings
    pub fn with_diversity(mut self, config: DiversityConfig) -> Self {
        self.diversity_config = config;
        self
    }

    /// Generate a single document
    pub async fn generate_one(&self, index: usize) -> Result<BeirDocument> {
        let doc_id = format!("doc_{:06}", index + 1);

        if self.dry_run {
            return Ok(BeirDocument {
                id: doc_id,
                title: format!("[DRY RUN] {} document #{}", self.topic_config.name, index + 1),
                text: format!(
                    "[DRY RUN] Would generate a {}-{} word {} document",
                    self.topic_config.min_words,
                    self.topic_config.max_words,
                    self.topic_config.name
                ),
            });
        }

        let system_prompt = document_system_prompt(&self.topic_config.name);
        let user_prompt = document_user_prompt(&self.topic_config, index);

        let text = self
            .provider
            .generate_document(&system_prompt, &user_prompt)
            .await?;

        // Extract title from first line if it looks like a title, otherwise generate one
        let (title, text) = extract_title_and_text(&text, &self.topic_config.name, index);

        Ok(BeirDocument {
            id: doc_id,
            title,
            text,
        })
    }

    /// Generate a document idea/title for two-phase generation
    async fn generate_idea(
        &self,
        existing_ideas: &[String],
        category: Option<&str>,
    ) -> Result<String> {
        let prompt = document_idea_prompt(&self.topic_config, existing_ideas, category);
        self.provider.generate_query(&prompt).await
    }

    /// Expand an idea into a full document
    async fn expand_idea(&self, idea: &str, index: usize) -> Result<BeirDocument> {
        let doc_id = format!("doc_{:06}", index + 1);

        let system_prompt = document_system_prompt(&self.topic_config.name);
        let user_prompt = document_expand_prompt(&self.topic_config, idea);

        let text = self
            .provider
            .generate_document(&system_prompt, &user_prompt)
            .await?;

        // Use idea as title, or extract from generated text
        let (title, text) = if text.lines().next().map(|l| l.trim().starts_with('#')).unwrap_or(false) {
            extract_title_and_text(&text, &self.topic_config.name, index)
        } else {
            (idea.to_string(), text)
        };

        Ok(BeirDocument { id: doc_id, title, text })
    }

    /// Get category for a document index (cycles through categories if provided)
    fn get_category(&self, index: usize) -> Option<&str> {
        self.diversity_config.categories.as_ref().map(|cats| {
            cats[index % cats.len()].as_str()
        })
    }

    /// Generate all documents with progress tracking (sequential)
    pub async fn generate_all(
        &self,
        count: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
    ) -> Result<Vec<BeirDocument>> {
        self.generate_all_concurrent(count, output_path, state, checkpoint_mgr, 1)
            .await
    }

    /// Generate all documents with concurrent execution and diversity checking
    pub async fn generate_all_concurrent(
        &self,
        count: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
        concurrency: usize,
    ) -> Result<Vec<BeirDocument>> {
        match self.diversity_config.mode {
            DocDiversity::None => {
                self.generate_all_basic(count, output_path, state, checkpoint_mgr, concurrency)
                    .await
            }
            DocDiversity::TwoPhase => {
                self.generate_all_two_phase(count, output_path, state, checkpoint_mgr, concurrency)
                    .await
            }
            DocDiversity::Embedding => {
                self.generate_all_embedding(count, output_path, state, checkpoint_mgr, concurrency)
                    .await
            }
        }
    }

    /// Basic generation without diversity checking (original behavior)
    async fn generate_all_basic(
        &self,
        count: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
        concurrency: usize,
    ) -> Result<Vec<BeirDocument>> {
        let mut documents = Vec::new();
        let concurrency = concurrency.max(1);

        let pb = ProgressBar::new(count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} documents ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Collect indices that need to be generated
        let mut pending_indices: Vec<usize> = Vec::new();
        for i in 0..count {
            let doc_id = format!("doc_{:06}", i + 1);
            if state.is_document_completed(&doc_id) {
                pb.inc(1);
            } else {
                pending_indices.push(i);
            }
        }

        if pending_indices.is_empty() {
            pb.finish_with_message("Document generation complete (all cached)");
            info!("All {} documents already generated", count);
            return Ok(documents);
        }

        // Generate documents concurrently in batches
        let semaphore = Arc::new(Semaphore::new(concurrency));

        // Process in chunks to allow checkpointing
        for chunk in pending_indices.chunks(concurrency * 2) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|&idx| {
                    let sem = semaphore.clone();
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = self.generate_one(idx).await;
                        (idx, result)
                    }
                })
                .collect();

            let results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            // Sort by index to maintain order for consistent output
            let mut sorted_results: Vec<_> = results.into_iter().collect();
            sorted_results.sort_by_key(|(idx, _)| *idx);

            for (idx, result) in sorted_results {
                let doc = result?;
                let doc_id = format!("doc_{:06}", idx + 1);

                // Append to file
                append_document(output_path, &doc)?;

                // Update progress
                state.mark_document_completed(&doc_id);
                checkpoint_mgr.record_and_maybe_save(state)?;

                documents.push(doc);
                pb.inc(1);
            }
        }

        pb.finish_with_message("Document generation complete");
        info!("Generated {} documents", documents.len());

        Ok(documents)
    }

    /// Two-phase generation: generate ideas first, then expand
    async fn generate_all_two_phase(
        &self,
        count: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
        concurrency: usize,
    ) -> Result<Vec<BeirDocument>> {
        let mut documents = Vec::new();
        let concurrency = concurrency.max(1);

        info!("Two-phase document generation: generating {} unique ideas first...", count);

        // Phase 1: Generate unique ideas
        let idea_tracker = IdeaTracker::new();
        let mut ideas: Vec<(usize, String)> = Vec::new(); // (index, idea)

        let pb_ideas = ProgressBar::new(count as u64);
        pb_ideas.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ideas ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Check for already completed documents
        let mut pending_indices: Vec<usize> = Vec::new();
        for i in 0..count {
            let doc_id = format!("doc_{:06}", i + 1);
            if state.is_document_completed(&doc_id) {
                pb_ideas.inc(1);
            } else {
                pending_indices.push(i);
            }
        }

        if pending_indices.is_empty() {
            pb_ideas.finish_with_message("All documents already generated");
            info!("All {} documents already generated", count);
            return Ok(documents);
        }

        const MAX_IDEA_RETRIES: usize = 10;

        // Generate ideas sequentially to maintain exclusion list
        for &idx in &pending_indices {
            let category = self.get_category(idx);
            let mut retries = 0;

            loop {
                let existing = idea_tracker.get_ideas();
                let idea = self.generate_idea(&existing, category).await?;

                // Check uniqueness with string similarity
                if idea_tracker.is_unique_enough(&idea, self.diversity_config.threshold) {
                    idea_tracker.register(&idea);
                    ideas.push((idx, idea));
                    pb_ideas.inc(1);
                    break;
                }

                retries += 1;
                if retries >= MAX_IDEA_RETRIES {
                    warn!("Could not generate unique idea after {} retries for index {}, using last attempt", MAX_IDEA_RETRIES, idx);
                    idea_tracker.register(&idea);
                    ideas.push((idx, idea));
                    pb_ideas.inc(1);
                    break;
                }
            }
        }

        pb_ideas.finish_with_message("Idea generation complete");
        info!("Generated {} unique document ideas", ideas.len());

        // Phase 2: Expand ideas into full documents
        let pb_docs = ProgressBar::new(ideas.len() as u64);
        pb_docs.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} documents ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let semaphore = Arc::new(Semaphore::new(concurrency));

        for chunk in ideas.chunks(concurrency * 2) {
            let futures: Vec<_> = chunk
                .iter()
                .map(|(idx, idea)| {
                    let sem = semaphore.clone();
                    let idea = idea.clone();
                    let idx = *idx;
                    async move {
                        let _permit = sem.acquire().await.unwrap();
                        let result = self.expand_idea(&idea, idx).await;
                        (idx, result)
                    }
                })
                .collect();

            let results: Vec<_> = stream::iter(futures)
                .buffer_unordered(concurrency)
                .collect()
                .await;

            let mut sorted_results: Vec<_> = results.into_iter().collect();
            sorted_results.sort_by_key(|(idx, _)| *idx);

            for (idx, result) in sorted_results {
                let doc = result?;
                let doc_id = format!("doc_{:06}", idx + 1);

                append_document(output_path, &doc)?;
                state.mark_document_completed(&doc_id);
                checkpoint_mgr.record_and_maybe_save(state)?;

                documents.push(doc);
                pb_docs.inc(1);
            }
        }

        pb_docs.finish_with_message("Document generation complete");
        info!("Generated {} documents via two-phase", documents.len());

        Ok(documents)
    }

    /// Embedding-based generation: reject docs too similar to existing
    async fn generate_all_embedding(
        &self,
        count: usize,
        output_path: &Path,
        state: &mut ProgressState,
        checkpoint_mgr: &mut CheckpointManager,
        _concurrency: usize, // TODO: Could batch embedding checks in the future
    ) -> Result<Vec<BeirDocument>> {
        let mut documents = Vec::new();

        // Create embedding client
        let embedding_url = self.diversity_config.embedding_url.as_deref()
            .unwrap_or(&self.provider.base_url());
        let embedding_model = self.diversity_config.embedding_model.as_deref()
            .unwrap_or(&self.provider.model());

        let embedding_client = EmbeddingClient::new(embedding_url, embedding_model);
        let diversity_checker = EmbeddingDiversityChecker::new(
            embedding_client,
            self.diversity_config.threshold,
        );

        info!("Embedding-based document generation with threshold {:.2}", self.diversity_config.threshold);

        let pb = ProgressBar::new(count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} documents ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        // Collect pending indices
        let mut pending_indices: Vec<usize> = Vec::new();
        for i in 0..count {
            let doc_id = format!("doc_{:06}", i + 1);
            if state.is_document_completed(&doc_id) {
                pb.inc(1);
            } else {
                pending_indices.push(i);
            }
        }

        if pending_indices.is_empty() {
            pb.finish_with_message("Document generation complete (all cached)");
            return Ok(documents);
        }

        const MAX_EMBEDDING_RETRIES: usize = 5;

        // Generate documents one at a time to maintain embedding state
        // (Could be optimized with batching, but simpler for now)
        for &idx in &pending_indices {
            let mut retries = 0;

            loop {
                let doc = self.generate_one(idx).await?;
                let doc_text = format!("{} {}", doc.title, doc.text);

                // Check diversity
                match diversity_checker.check_and_register(&doc_text).await {
                    Ok(true) => {
                        // Unique enough
                        let doc_id = format!("doc_{:06}", idx + 1);
                        append_document(output_path, &doc)?;
                        state.mark_document_completed(&doc_id);
                        checkpoint_mgr.record_and_maybe_save(state)?;
                        documents.push(doc);
                        pb.inc(1);
                        break;
                    }
                    Ok(false) => {
                        retries += 1;
                        if retries >= MAX_EMBEDDING_RETRIES {
                            warn!("Document {} too similar after {} retries, accepting anyway", idx, retries);
                            let doc_id = format!("doc_{:06}", idx + 1);
                            append_document(output_path, &doc)?;
                            state.mark_document_completed(&doc_id);
                            checkpoint_mgr.record_and_maybe_save(state)?;
                            documents.push(doc);
                            pb.inc(1);
                            break;
                        }
                        warn!("Document {} too similar (retry {}/{})", idx, retries, MAX_EMBEDDING_RETRIES);
                    }
                    Err(e) => {
                        warn!("Embedding check failed: {}, accepting document", e);
                        let doc_id = format!("doc_{:06}", idx + 1);
                        append_document(output_path, &doc)?;
                        state.mark_document_completed(&doc_id);
                        checkpoint_mgr.record_and_maybe_save(state)?;
                        documents.push(doc);
                        pb.inc(1);
                        break;
                    }
                }
            }
        }

        pb.finish_with_message("Document generation complete");
        info!("Generated {} documents with embedding diversity", documents.len());

        Ok(documents)
    }
}

/// Extract title from document text or generate a placeholder
fn extract_title_and_text(text: &str, topic: &str, index: usize) -> (String, String) {
    let lines: Vec<&str> = text.lines().collect();

    if lines.is_empty() {
        return (
            format!("{} Document #{}", capitalize(topic), index + 1),
            String::new(),
        );
    }

    let first_line = lines[0].trim();

    // Check if first line looks like a title (short, possibly with # prefix)
    let is_title = first_line.len() < 100
        && !first_line.ends_with('.')
        && !first_line.ends_with(',')
        && (first_line.starts_with('#')
            || first_line.starts_with("Title:")
            || lines.len() > 1 && lines[1].trim().is_empty());

    if is_title {
        let title = first_line
            .trim_start_matches('#')
            .trim_start_matches("Title:")
            .trim();

        let remaining_text = lines[1..].join("\n").trim().to_string();
        (title.to_string(), remaining_text)
    } else {
        // Generate a title from first few words
        let words: Vec<&str> = first_line.split_whitespace().take(6).collect();
        let title = if words.len() >= 3 {
            format!("{}...", words.join(" "))
        } else {
            format!("{} Document #{}", capitalize(topic), index + 1)
        };

        (title, text.to_string())
    }
}

fn capitalize(s: &str) -> String {
    let mut c = s.chars();
    match c.next() {
        None => String::new(),
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_title_with_hash() {
        let text = "# My Recipe\n\nIngredients:\n- Eggs";
        let (title, body) = extract_title_and_text(text, "recipes", 0);
        assert_eq!(title, "My Recipe");
        assert!(body.contains("Ingredients"));
    }

    #[test]
    fn test_extract_title_no_title() {
        let text = "This is a long paragraph that doesn't have a clear title.";
        let (title, body) = extract_title_and_text(text, "notes", 0);
        assert!(title.contains("...") || title.contains("Notes"));
        assert_eq!(body, text);
    }
}
