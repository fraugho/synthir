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
    pub mixed: usize,
}

/// Organizes the output directory structure
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

    /// Get queries directory for a query type
    pub fn queries_dir(&self, query_type: QueryType) -> PathBuf {
        self.topic_dir().join(query_type.as_str())
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

    /// Get combined directory (everything in one place)
    pub fn combined_dir(&self) -> PathBuf {
        self.topic_dir().join("combined")
    }

    /// Get hard negatives file path
    pub fn hard_negatives_path(&self) -> PathBuf {
        self.hard_negatives_merged_dir().join("hard_negatives.jsonl")
    }

    /// Get metadata file path
    pub fn metadata_path(&self) -> PathBuf {
        self.base_dir.join("metadata.json")
    }

    /// Create the full directory structure for all query types
    pub fn create_structure(&self) -> Result<()> {
        let mut all_types = QueryType::all_types();
        all_types.push(QueryType::Mixed);
        self.create_structure_for_types(&all_types, true, true)
    }

    /// Create directory structure for specific query types only
    pub fn create_structure_for_types(&self, query_types: &[QueryType], include_hard_negatives: bool, include_merged: bool) -> Result<()> {
        // Create topic directory
        fs::create_dir_all(self.topic_dir())?;

        // Create only the requested query type directories
        for query_type in query_types {
            fs::create_dir_all(self.queries_dir(*query_type))?;
        }

        // Create merged/combined directories only if requested
        if include_merged {
            fs::create_dir_all(self.general_merged_dir())?;
            if include_hard_negatives {
                fs::create_dir_all(self.hard_negatives_merged_dir())?;
            }
            fs::create_dir_all(self.combined_dir())?;
        }

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
