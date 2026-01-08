use crate::config::{GenerationConfig, QueryType};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

/// Current phase of generation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GenerationPhase {
    DocumentGeneration,
    QueryGeneration,
    RelevanceScoring,
    HardNegativeMining,
    OutputOrganization,
    Complete,
}

impl std::fmt::Display for GenerationPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GenerationPhase::DocumentGeneration => write!(f, "Document Generation"),
            GenerationPhase::QueryGeneration => write!(f, "Query Generation"),
            GenerationPhase::RelevanceScoring => write!(f, "Relevance Scoring"),
            GenerationPhase::HardNegativeMining => write!(f, "Hard Negative Mining"),
            GenerationPhase::OutputOrganization => write!(f, "Output Organization"),
            GenerationPhase::Complete => write!(f, "Complete"),
        }
    }
}

/// Progress state that gets persisted
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProgressState {
    /// Current phase
    pub phase: GenerationPhase,

    /// Topic being generated
    pub topic: String,

    /// IDs of completed documents
    pub completed_documents: HashSet<String>,

    /// IDs of completed queries per type
    pub completed_queries: HashMap<QueryType, HashSet<String>>,

    /// Number of completed relevance scoring pairs
    pub completed_relevance_pairs: usize,

    /// Number of completed hard negative validations
    pub completed_hard_negatives: usize,

    /// Original generation config
    pub config: GenerationConfig,
}

impl ProgressState {
    pub fn new(config: GenerationConfig) -> Self {
        Self {
            phase: GenerationPhase::DocumentGeneration,
            topic: config.topic.clone(),
            completed_documents: HashSet::new(),
            completed_queries: HashMap::new(),
            completed_relevance_pairs: 0,
            completed_hard_negatives: 0,
            config,
        }
    }

    /// Get progress file path for an output directory
    pub fn progress_file_path(output_dir: &Path) -> PathBuf {
        output_dir.join("progress.json")
    }

    /// Load progress from file if it exists
    pub fn load(output_dir: &Path) -> Result<Option<Self>> {
        let path = Self::progress_file_path(output_dir);
        if !path.exists() {
            return Ok(None);
        }

        let content = fs::read_to_string(&path)
            .with_context(|| format!("Failed to read progress file: {}", path.display()))?;

        let state: Self = serde_json::from_str(&content)
            .with_context(|| "Failed to parse progress file")?;

        info!("Loaded progress from {}", path.display());
        info!("  Phase: {}", state.phase);
        info!("  Completed documents: {}", state.completed_documents.len());

        Ok(Some(state))
    }

    /// Save progress to file (atomic write)
    pub fn save(&self, output_dir: &Path) -> Result<()> {
        fs::create_dir_all(output_dir)?;

        let path = Self::progress_file_path(output_dir);
        let temp_path = path.with_extension("json.tmp");

        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize progress")?;

        fs::write(&temp_path, content)
            .with_context(|| format!("Failed to write temp progress file: {}", temp_path.display()))?;

        fs::rename(&temp_path, &path)
            .with_context(|| format!("Failed to rename progress file to: {}", path.display()))?;

        Ok(())
    }

    /// Mark a document as completed
    pub fn mark_document_completed(&mut self, doc_id: &str) {
        self.completed_documents.insert(doc_id.to_string());
    }

    /// Mark a query as completed
    pub fn mark_query_completed(&mut self, query_type: QueryType, query_id: &str) {
        self.completed_queries
            .entry(query_type)
            .or_default()
            .insert(query_id.to_string());
    }

    /// Check if a document is completed
    pub fn is_document_completed(&self, doc_id: &str) -> bool {
        self.completed_documents.contains(doc_id)
    }

    /// Check if a query is completed
    pub fn is_query_completed(&self, query_type: QueryType, query_id: &str) -> bool {
        self.completed_queries
            .get(&query_type)
            .map(|set| set.contains(query_id))
            .unwrap_or(false)
    }

    /// Advance to next phase
    pub fn advance_phase(&mut self) {
        self.phase = match self.phase {
            GenerationPhase::DocumentGeneration => GenerationPhase::QueryGeneration,
            GenerationPhase::QueryGeneration => GenerationPhase::RelevanceScoring,
            GenerationPhase::RelevanceScoring => GenerationPhase::HardNegativeMining,
            GenerationPhase::HardNegativeMining => GenerationPhase::OutputOrganization,
            GenerationPhase::OutputOrganization => GenerationPhase::Complete,
            GenerationPhase::Complete => GenerationPhase::Complete,
        };
    }

    /// Get count of completed queries for a type
    pub fn completed_query_count(&self, query_type: QueryType) -> usize {
        self.completed_queries
            .get(&query_type)
            .map(|set| set.len())
            .unwrap_or(0)
    }
}

/// Checkpoint manager handles periodic saving
pub struct CheckpointManager {
    output_dir: PathBuf,
    batch_size: usize,
    items_since_save: usize,
}

impl CheckpointManager {
    pub fn new(output_dir: PathBuf, batch_size: usize) -> Self {
        Self {
            output_dir,
            batch_size,
            items_since_save: 0,
        }
    }

    /// Record an item completion and save if batch is full
    pub fn record_and_maybe_save(&mut self, state: &ProgressState) -> Result<()> {
        self.items_since_save += 1;

        if self.items_since_save >= self.batch_size {
            state.save(&self.output_dir)?;
            self.items_since_save = 0;
        }

        Ok(())
    }

    /// Force save regardless of batch
    pub fn force_save(&mut self, state: &ProgressState) -> Result<()> {
        state.save(&self.output_dir)?;
        self.items_since_save = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_progress_save_load() {
        let dir = tempdir().unwrap();
        let config = GenerationConfig::default();
        let mut state = ProgressState::new(config);

        state.mark_document_completed("doc_001");
        state.mark_document_completed("doc_002");
        state.mark_query_completed(QueryType::Natural, "q_001");

        state.save(dir.path()).unwrap();

        let loaded = ProgressState::load(dir.path()).unwrap().unwrap();
        assert!(loaded.is_document_completed("doc_001"));
        assert!(loaded.is_document_completed("doc_002"));
        assert!(loaded.is_query_completed(QueryType::Natural, "q_001"));
        assert!(!loaded.is_query_completed(QueryType::Keyword, "q_001"));
    }
}
