use crate::config::QueryType;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

/// Metadata about the generated dataset
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetadata {
    pub topic: String,
    pub document_count: usize,
    pub query_counts: QueryCounts,
    pub generation_timestamp: String,
    pub model_used: String,
    pub has_hard_negatives: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryCounts {
    pub natural: usize,
    pub keyword: usize,
    pub academic: usize,
    pub complex: usize,
    pub semantic: usize,
    pub basic: usize,
    pub mixed: usize,
}

/// Organizes the output directory structure
///
/// Simplified flat structure:
/// ```
/// dataset_name/
/// ├── corpus.jsonl
/// ├── queries.jsonl
/// ├── qrels.tsv
/// └── metadata.json
/// ```
pub struct OutputOrganizer {
    base_dir: PathBuf,
    topic: String,
    /// If true, output goes directly to base_dir without topic subdirectory
    flat_output: bool,
}

impl OutputOrganizer {
    pub fn new(base_dir: PathBuf, topic: String) -> Self {
        Self { base_dir, topic, flat_output: false }
    }

    /// Create organizer that outputs directly to base_dir without topic subdirectory
    pub fn new_flat(base_dir: PathBuf, topic: String) -> Self {
        Self { base_dir, topic, flat_output: true }
    }

    /// Get the topic directory
    pub fn topic_dir(&self) -> PathBuf {
        if self.flat_output {
            self.base_dir.clone()
        } else {
            self.base_dir.join(&self.topic)
        }
    }

    /// Get the corpus file path
    pub fn corpus_path(&self) -> PathBuf {
        self.topic_dir().join("corpus.jsonl")
    }

    /// Get queries file path (all queries in one file, same directory as corpus)
    pub fn queries_path(&self, _query_type: QueryType) -> PathBuf {
        self.topic_dir().join("queries.jsonl")
    }

    /// Get the single queries file path
    pub fn queries_file(&self) -> PathBuf {
        self.topic_dir().join("queries.jsonl")
    }

    /// Get qrels file path (same directory as corpus)
    pub fn qrels_path(&self, _query_type: QueryType) -> PathBuf {
        self.topic_dir().join("qrels.tsv")
    }

    /// Get the single qrels file path
    pub fn qrels_file(&self) -> PathBuf {
        self.topic_dir().join("qrels.tsv")
    }

    /// Get metadata file path
    pub fn metadata_path(&self) -> PathBuf {
        self.topic_dir().join("metadata.json")
    }

    /// Create the directory structure (simplified - just the topic directory)
    pub fn create_structure(&self) -> Result<()> {
        fs::create_dir_all(self.topic_dir())?;
        Ok(())
    }

    /// Create directory structure (simplified - ignores query types, just creates topic dir)
    pub fn create_structure_for_types(&self, _query_types: &[QueryType], _include_hard_negatives: bool, _include_merged: bool) -> Result<()> {
        fs::create_dir_all(self.topic_dir())?;
        Ok(())
    }

    /// Write metadata file
    pub fn write_metadata(&self, metadata: &DatasetMetadata) -> Result<()> {
        let content = serde_json::to_string_pretty(metadata)?;
        fs::write(self.metadata_path(), content)?;
        Ok(())
    }

    /// Get all output paths for dry-run display
    pub fn all_paths(&self) -> Vec<PathBuf> {
        vec![
            self.corpus_path(),
            self.queries_file(),
            self.qrels_file(),
            self.metadata_path(),
        ]
    }

    // Legacy methods for backwards compatibility - now point to main files

    /// Get queries directory for a query type (legacy - returns topic dir)
    #[allow(dead_code)]
    pub fn queries_dir(&self, _query_type: QueryType) -> PathBuf {
        self.topic_dir()
    }

    /// Get merged directory (legacy - returns topic dir)
    #[allow(dead_code)]
    pub fn merged_dir(&self) -> PathBuf {
        self.topic_dir()
    }

    /// Get general merged directory (legacy - returns topic dir)
    pub fn general_merged_dir(&self) -> PathBuf {
        self.topic_dir()
    }

    /// Get merged directory with hard negatives (legacy - returns topic dir)
    pub fn hard_negatives_merged_dir(&self) -> PathBuf {
        self.topic_dir()
    }

    /// Get combined directory (legacy - returns topic dir)
    pub fn combined_dir(&self) -> PathBuf {
        self.topic_dir()
    }

    /// Get hard negatives file path
    #[allow(dead_code)]
    pub fn hard_negatives_path(&self) -> PathBuf {
        self.topic_dir().join("hard_negatives.jsonl")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_output_structure() {
        let dir = tempdir().unwrap();
        let organizer = OutputOrganizer::new(dir.path().to_path_buf(), "recipes".to_string());

        organizer.create_structure().unwrap();

        assert!(organizer.topic_dir().exists());
        assert!(organizer.queries_dir(QueryType::Natural).exists());
        assert!(organizer.general_merged_dir().exists());
    }
}
