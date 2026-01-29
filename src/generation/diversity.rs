use crate::llm::EmbeddingClient;
use anyhow::Result;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use tracing::{debug, info};

/// Cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot / (norm_a * norm_b)
    }
}

/// Tracks document ideas/titles for two-phase deduplication
#[derive(Clone)]
pub struct IdeaTracker {
    /// Normalized ideas (lowercase, trimmed)
    ideas: Arc<Mutex<HashSet<String>>>,
    /// Original ideas for display/prompts
    ideas_list: Arc<Mutex<Vec<String>>>,
}

impl IdeaTracker {
    pub fn new() -> Self {
        Self {
            ideas: Arc::new(Mutex::new(HashSet::new())),
            ideas_list: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Check if an idea is unique and register it
    /// Returns true if unique (newly registered)
    pub fn register(&self, idea: &str) -> bool {
        let normalized = idea.to_lowercase().trim().to_string();
        let mut ideas = self.ideas.lock().unwrap();
        if ideas.insert(normalized) {
            let mut list = self.ideas_list.lock().unwrap();
            list.push(idea.to_string());
            true
        } else {
            false
        }
    }

    /// Get all registered ideas (for exclusion prompts)
    pub fn get_ideas(&self) -> Vec<String> {
        let list = self.ideas_list.lock().unwrap();
        list.clone()
    }

    /// Check similarity against existing ideas (simple string matching)
    /// Returns true if the idea is sufficiently unique
    pub fn is_unique_enough(&self, idea: &str, threshold: f32) -> bool {
        let normalized = idea.to_lowercase();
        let ideas = self.ideas.lock().unwrap();

        for existing in ideas.iter() {
            // Simple Jaccard similarity on words
            let new_words: HashSet<&str> = normalized.split_whitespace().collect();
            let old_words: HashSet<&str> = existing.split_whitespace().collect();

            if new_words.is_empty() || old_words.is_empty() {
                continue;
            }

            let intersection = new_words.intersection(&old_words).count();
            let union = new_words.union(&old_words).count();

            let jaccard = intersection as f32 / union as f32;
            if jaccard > threshold {
                debug!(
                    "Idea '{}' too similar to '{}' (jaccard={:.2})",
                    idea, existing, jaccard
                );
                return false;
            }
        }

        true
    }
}

impl Default for IdeaTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Embedding-based diversity checker
pub struct EmbeddingDiversityChecker {
    client: EmbeddingClient,
    embeddings: Arc<Mutex<Vec<Vec<f32>>>>,
    threshold: f32,
}

impl EmbeddingDiversityChecker {
    pub fn new(client: EmbeddingClient, threshold: f32) -> Self {
        Self {
            client,
            embeddings: Arc::new(Mutex::new(Vec::new())),
            threshold,
        }
    }

    /// Check if text is diverse enough from existing items
    /// Returns Ok(true) if unique, Ok(false) if too similar
    pub async fn check_and_register(&self, text: &str) -> Result<bool> {
        let embedding = self.client.embed_one(text).await?;

        let embeddings = self.embeddings.lock().unwrap();

        // Check against all existing embeddings
        for (idx, existing) in embeddings.iter().enumerate() {
            let sim = cosine_similarity(&embedding, existing);
            if sim > self.threshold {
                debug!(
                    "Text too similar to item {} (cosine={:.3}, threshold={:.3})",
                    idx, sim, self.threshold
                );
                return Ok(false);
            }
        }

        drop(embeddings);

        // Register the new embedding
        let mut embeddings = self.embeddings.lock().unwrap();
        embeddings.push(embedding);

        Ok(true)
    }

    /// Pre-populate with existing items (e.g., from checkpoint)
    pub async fn preload(&self, texts: &[String]) -> Result<()> {
        if texts.is_empty() {
            return Ok(());
        }

        info!("Preloading {} embeddings for diversity checking...", texts.len());

        // Embed in batches
        let batch_size = 32;
        for chunk in texts.chunks(batch_size) {
            let chunk_vec: Vec<String> = chunk.to_vec();
            let batch_embeddings = self.client.embed_batch(&chunk_vec).await?;

            let mut embeddings = self.embeddings.lock().unwrap();
            embeddings.extend(batch_embeddings);
        }

        Ok(())
    }

    /// Get current count of tracked items
    pub fn count(&self) -> usize {
        self.embeddings.lock().unwrap().len()
    }
}

/// Query diversity checker using embeddings
pub struct QueryDiversityChecker {
    inner: EmbeddingDiversityChecker,
    /// Also track exact strings
    exact_queries: Arc<Mutex<HashSet<String>>>,
}

impl QueryDiversityChecker {
    pub fn new(client: EmbeddingClient, threshold: f32) -> Self {
        Self {
            inner: EmbeddingDiversityChecker::new(client, threshold),
            exact_queries: Arc::new(Mutex::new(HashSet::new())),
        }
    }

    /// Check if query is diverse enough (both exact and semantic)
    pub async fn check_and_register(&self, query: &str) -> Result<bool> {
        let normalized = query.to_lowercase().trim().to_string();

        // First check exact match
        {
            let exact = self.exact_queries.lock().unwrap();
            if exact.contains(&normalized) {
                debug!("Query '{}' is exact duplicate", query);
                return Ok(false);
            }
        }

        // Then check semantic similarity
        if !self.inner.check_and_register(query).await? {
            return Ok(false);
        }

        // Register exact string
        let mut exact = self.exact_queries.lock().unwrap();
        exact.insert(normalized);

        Ok(true)
    }

    /// Preload existing queries
    pub async fn preload(&self, queries: &[String]) -> Result<()> {
        // Register exact strings
        {
            let mut exact = self.exact_queries.lock().unwrap();
            for q in queries {
                exact.insert(q.to_lowercase().trim().to_string());
            }
        }

        // Preload embeddings
        self.inner.preload(queries).await
    }

    pub fn count(&self) -> usize {
        self.inner.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine_similarity(&a, &c) - 0.0).abs() < 0.001);

        let d = vec![1.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &d);
        assert!(sim > 0.7 && sim < 0.8);
    }

    #[test]
    fn test_idea_tracker() {
        let tracker = IdeaTracker::new();

        assert!(tracker.register("Spicy Thai Curry Recipe"));
        assert!(!tracker.register("spicy thai curry recipe")); // Case insensitive duplicate

        assert!(tracker.register("Italian Pasta Carbonara"));

        let ideas = tracker.get_ideas();
        assert_eq!(ideas.len(), 2);
    }

    #[test]
    fn test_idea_uniqueness() {
        let tracker = IdeaTracker::new();
        tracker.register("spicy chicken curry recipe");

        // Very similar idea
        assert!(!tracker.is_unique_enough("spicy beef curry recipe", 0.5));

        // More different idea
        assert!(tracker.is_unique_enough("italian pasta carbonara", 0.5));
    }
}
