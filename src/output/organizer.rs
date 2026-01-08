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
    pub mixed: usize,
}

/// Organizes the output directory structure
pub struct OutputOrganizer {
    base_dir: PathBuf,
    topic: String,
}

impl OutputOrganizer {
    pub fn new(base_dir: PathBuf, topic: String) -> Self {
        Self { base_dir, topic }
    }

    /// Get the topic directory
    pub fn topic_dir(&self) -> PathBuf {
        self.base_dir.join(&self.topic)
    }

    /// Get the corpus file path
    pub fn corpus_path(&self) -> PathBuf {
        self.topic_dir().join("corpus.jsonl")
    }

    /// Get queries directory for a query type
    pub fn queries_dir(&self, query_type: QueryType) -> PathBuf {
        self.topic_dir().join("queries").join(query_type.as_str())
    }

    /// Get queries file path for a query type
    pub fn queries_path(&self, query_type: QueryType) -> PathBuf {
        self.queries_dir(query_type).join("queries.jsonl")
    }

    /// Get qrels file path for a query type
    pub fn qrels_path(&self, query_type: QueryType) -> PathBuf {
        self.queries_dir(query_type).join("qrels.tsv")
    }

    /// Get merged directory
    pub fn merged_dir(&self) -> PathBuf {
        self.topic_dir().join("merged")
    }

    /// Get general merged directory (without hard negatives)
    pub fn general_merged_dir(&self) -> PathBuf {
        self.merged_dir().join("general")
    }

    /// Get merged directory with hard negatives
    pub fn hard_negatives_merged_dir(&self) -> PathBuf {
        self.merged_dir().join("with-hard-negatives")
    }

    /// Get hard negatives file path
    pub fn hard_negatives_path(&self) -> PathBuf {
        self.hard_negatives_merged_dir().join("hard_negatives.jsonl")
    }

    /// Get metadata file path
    pub fn metadata_path(&self) -> PathBuf {
        self.base_dir.join("metadata.json")
    }

    /// Create the full directory structure
    pub fn create_structure(&self) -> Result<()> {
        // Create topic directory
        fs::create_dir_all(self.topic_dir())?;

        // Create query type directories
        for query_type in QueryType::all_types() {
            fs::create_dir_all(self.queries_dir(query_type))?;
        }
        fs::create_dir_all(self.queries_dir(QueryType::Mixed))?;

        // Create merged directories
        fs::create_dir_all(self.general_merged_dir())?;
        fs::create_dir_all(self.hard_negatives_merged_dir())?;

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
        let mut paths = vec![
            self.corpus_path(),
            self.metadata_path(),
        ];

        for query_type in QueryType::all_types() {
            paths.push(self.queries_path(query_type));
            paths.push(self.qrels_path(query_type));
        }
        paths.push(self.queries_path(QueryType::Mixed));
        paths.push(self.qrels_path(QueryType::Mixed));

        paths.push(self.general_merged_dir().join("queries.jsonl"));
        paths.push(self.general_merged_dir().join("qrels.tsv"));
        paths.push(self.hard_negatives_merged_dir().join("queries.jsonl"));
        paths.push(self.hard_negatives_merged_dir().join("qrels.tsv"));
        paths.push(self.hard_negatives_path());

        paths
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
