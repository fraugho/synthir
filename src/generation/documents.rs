use crate::checkpoint::{CheckpointManager, ProgressState};
use crate::llm::{document_system_prompt, document_user_prompt, LLMProvider};
use crate::output::{append_document, BeirDocument};
use crate::topics::TopicConfig;
use anyhow::Result;
use futures::stream::{self, StreamExt};
use indicatif::{ProgressBar, ProgressStyle};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Semaphore;
use tracing::info;

/// Generate documents for a topic
pub struct DocumentGenerator<'a> {
    provider: &'a LLMProvider,
    topic_config: TopicConfig,
    dry_run: bool,
}

impl<'a> DocumentGenerator<'a> {
    pub fn new(provider: &'a LLMProvider, topic_config: TopicConfig, dry_run: bool) -> Self {
        Self {
            provider,
            topic_config,
            dry_run,
        }
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

    /// Generate all documents with concurrent execution
    pub async fn generate_all_concurrent(
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
