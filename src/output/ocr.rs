use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;

use super::{BeirDocument, BeirQuery, Qrel};

/// OCR dataset format as used by GoodNotesOCR
/// - label.json: maps doc_id -> OCR text content
/// - queries.json: maps query_text -> [relevant_doc_ids]

/// Read corpus from OCR label.json format
/// Returns BeirDocument with id from key, text from value, empty title
pub fn read_ocr_corpus(path: &Path) -> Result<Vec<BeirDocument>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open OCR label file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let labels: HashMap<String, String> = serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse OCR label.json: {}", path.display()))?;

    let mut documents: Vec<BeirDocument> = labels
        .into_iter()
        .map(|(id, text)| BeirDocument {
            id,
            title: String::new(),
            text,
        })
        .collect();

    // Sort by ID for deterministic ordering
    documents.sort_by(|a, b| a.id.cmp(&b.id));

    Ok(documents)
}

/// OCR queries format: query_text -> [relevant_doc_ids]
pub type OcrQueries = HashMap<String, Vec<String>>;

/// Read queries and qrels from OCR queries.json format
/// Returns (queries, qrels) where queries have generated IDs
pub fn read_ocr_queries(path: &Path) -> Result<(Vec<BeirQuery>, Vec<Qrel>)> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open OCR queries file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let ocr_queries: OcrQueries = serde_json::from_reader(reader)
        .with_context(|| format!("Failed to parse OCR queries.json: {}", path.display()))?;

    let mut queries = Vec::new();
    let mut qrels = Vec::new();

    for (idx, (query_text, doc_ids)) in ocr_queries.into_iter().enumerate() {
        let query_id = format!("q{}", idx + 1);

        queries.push(BeirQuery {
            id: query_id.clone(),
            text: query_text,
        });

        for doc_id in doc_ids {
            qrels.push(Qrel {
                query_id: query_id.clone(),
                doc_id,
                score: 1, // OCR format uses binary relevance
            });
        }
    }

    // Sort queries by ID for deterministic ordering
    queries.sort_by(|a, b| a.id.cmp(&b.id));

    Ok((queries, qrels))
}

/// Write queries and qrels in OCR queries.json format
pub fn write_ocr_queries(path: &Path, queries: &[BeirQuery], qrels: &[Qrel]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    // Build query_text -> [doc_ids] mapping
    let mut ocr_queries: HashMap<String, Vec<String>> = HashMap::new();

    // First, create a map from query_id to query_text
    let query_text_map: HashMap<&str, &str> = queries
        .iter()
        .map(|q| (q.id.as_str(), q.text.as_str()))
        .collect();

    // Group qrels by query
    for qrel in qrels {
        if let Some(query_text) = query_text_map.get(qrel.query_id.as_str()) {
            ocr_queries
                .entry(query_text.to_string())
                .or_default()
                .push(qrel.doc_id.clone());
        }
    }

    let file = File::create(path)
        .with_context(|| format!("Failed to create OCR queries file: {}", path.display()))?;
    let writer = BufWriter::new(file);

    serde_json::to_writer_pretty(writer, &ocr_queries)?;

    Ok(())
}

/// Dataset format detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetFormat {
    /// BEIR format: corpus.jsonl, queries.jsonl, qrels/*.tsv
    Beir,
    /// OCR format: label.json, queries.json
    Ocr,
}

impl std::fmt::Display for DatasetFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DatasetFormat::Beir => write!(f, "beir"),
            DatasetFormat::Ocr => write!(f, "ocr"),
        }
    }
}

/// Information about an existing dataset's structure
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    pub format: DatasetFormat,
    pub root_dir: std::path::PathBuf,
    /// For BEIR: path to corpus.jsonl; For OCR: path to label.json
    pub corpus_path: std::path::PathBuf,
    /// For BEIR: path to queries.jsonl; For OCR: path to queries.json
    pub queries_path: Option<std::path::PathBuf>,
    /// Paths to qrels files (for BEIR: may have train.tsv, dev.tsv, test.tsv)
    pub qrels_paths: Vec<std::path::PathBuf>,
    /// Locale subdirectory if present (e.g., "en-us")
    pub locale: Option<String>,
}

/// Detect dataset format and gather structure info
pub fn detect_dataset(path: &Path) -> Result<DatasetInfo> {
    // Check if path is a directory
    if !path.is_dir() {
        anyhow::bail!("Dataset path must be a directory: {}", path.display());
    }

    // First check for locale subdirectory (common in eval datasets)
    let (effective_dir, locale) = if let Some(locale_dir) = find_locale_subdir(path) {
        let locale_name = locale_dir
            .file_name()
            .and_then(|n| n.to_str())
            .map(String::from);
        (locale_dir, locale_name)
    } else {
        (path.to_path_buf(), None)
    };

    // Check for OCR format (label.json)
    let label_path = effective_dir.join("label.json");
    if label_path.exists() {
        let queries_path = effective_dir.join("queries.json");
        return Ok(DatasetInfo {
            format: DatasetFormat::Ocr,
            root_dir: path.to_path_buf(),
            corpus_path: label_path,
            queries_path: if queries_path.exists() {
                Some(queries_path)
            } else {
                None
            },
            qrels_paths: vec![], // OCR embeds qrels in queries.json
            locale,
        });
    }

    // Check for BEIR format (corpus.jsonl)
    let corpus_path = effective_dir.join("corpus.jsonl");
    if corpus_path.exists() {
        let queries_path = effective_dir.join("queries.jsonl");
        let qrels_paths = find_qrels_files(&effective_dir);

        return Ok(DatasetInfo {
            format: DatasetFormat::Beir,
            root_dir: path.to_path_buf(),
            corpus_path,
            queries_path: if queries_path.exists() {
                Some(queries_path)
            } else {
                None
            },
            qrels_paths,
            locale,
        });
    }

    anyhow::bail!(
        "Could not detect dataset format. Expected label.json (OCR) or corpus.jsonl (BEIR) in {}",
        path.display()
    );
}

/// Find locale subdirectory (e.g., en-us)
fn find_locale_subdir(path: &Path) -> Option<std::path::PathBuf> {
    let entries = fs::read_dir(path).ok()?;

    for entry in entries.flatten() {
        let entry_path = entry.path();
        if entry_path.is_dir() {
            let name = entry.file_name();
            let name_str = name.to_str()?;
            // Common locale patterns
            if name_str.contains('-') && name_str.len() <= 10 {
                // Check if it contains corpus or label files
                if entry_path.join("corpus.jsonl").exists()
                    || entry_path.join("label.json").exists()
                {
                    return Some(entry_path);
                }
            }
        }
    }

    None
}

/// Find all qrels files in a BEIR dataset
fn find_qrels_files(path: &Path) -> Vec<std::path::PathBuf> {
    let mut qrels = Vec::new();

    // Check for qrels directory
    let qrels_dir = path.join("qrels");
    if qrels_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&qrels_dir) {
            for entry in entries.flatten() {
                let entry_path = entry.path();
                if entry_path.extension().map_or(false, |e| e == "tsv") {
                    qrels.push(entry_path);
                }
            }
        }
    }

    // Also check for qrels.tsv directly in the directory
    let direct_qrels = path.join("qrels.tsv");
    if direct_qrels.exists() {
        qrels.push(direct_qrels);
    }

    // Sort for deterministic ordering
    qrels.sort();

    qrels
}

/// Read corpus from either format based on detected info
pub fn read_corpus_auto(info: &DatasetInfo) -> Result<Vec<BeirDocument>> {
    match info.format {
        DatasetFormat::Beir => super::read_corpus(&info.corpus_path),
        DatasetFormat::Ocr => read_ocr_corpus(&info.corpus_path),
    }
}

/// Qrels split info for preserving train/dev/test ratios
#[derive(Debug, Clone)]
pub struct QrelsSplit {
    pub name: String, // "train", "dev", "test", or "qrels"
    pub path: std::path::PathBuf,
    pub count: usize,
}

/// Analyze qrels splits to determine ratios
pub fn analyze_qrels_splits(info: &DatasetInfo) -> Result<Vec<QrelsSplit>> {
    let mut splits = Vec::new();

    if info.format == DatasetFormat::Ocr {
        // OCR format has qrels embedded in queries.json
        if let Some(queries_path) = &info.queries_path {
            let (_, qrels) = read_ocr_queries(queries_path)?;
            splits.push(QrelsSplit {
                name: "queries".to_string(),
                path: queries_path.clone(),
                count: qrels.len(),
            });
        }
    } else {
        // BEIR format may have multiple splits
        for qrels_path in &info.qrels_paths {
            let qrels = super::read_qrels(qrels_path)?;
            let name = qrels_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("qrels")
                .to_string();
            splits.push(QrelsSplit {
                name,
                path: qrels_path.clone(),
                count: qrels.len(),
            });
        }
    }

    Ok(splits)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_ocr_corpus_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("label.json");

        let labels: HashMap<String, String> = [
            ("page_1.png".to_string(), "First page content".to_string()),
            ("page_2.png".to_string(), "Second page content".to_string()),
        ]
        .into_iter()
        .collect();

        let content = serde_json::to_string_pretty(&labels).unwrap();
        fs::write(&path, content).unwrap();

        let docs = read_ocr_corpus(&path).unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn test_ocr_queries_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("queries.json");

        let queries: OcrQueries = [
            (
                "test query".to_string(),
                vec!["page_1.png".to_string(), "page_2.png".to_string()],
            ),
        ]
        .into_iter()
        .collect();

        let content = serde_json::to_string_pretty(&queries).unwrap();
        fs::write(&path, content).unwrap();

        let (parsed_queries, qrels) = read_ocr_queries(&path).unwrap();
        assert_eq!(parsed_queries.len(), 1);
        assert_eq!(qrels.len(), 2);
    }
}
