use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// A document in BEIR corpus format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeirDocument {
    #[serde(rename = "_id")]
    pub id: String,
    pub title: String,
    pub text: String,
}

/// A query in BEIR format
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeirQuery {
    #[serde(rename = "_id")]
    pub id: String,
    pub text: String,
}

/// A relevance judgment (qrel)
#[derive(Debug, Clone)]
pub struct Qrel {
    pub query_id: String,
    pub doc_id: String,
    pub score: u16,
}

/// Write documents to a BEIR corpus.jsonl file
pub fn write_corpus(path: &Path, documents: &[BeirDocument]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(path)
        .with_context(|| format!("Failed to create corpus file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    for doc in documents {
        let line = serde_json::to_string(doc)?;
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write queries to a BEIR queries.jsonl file
pub fn write_queries(path: &Path, queries: &[BeirQuery]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(path)
        .with_context(|| format!("Failed to create queries file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    for query in queries {
        let line = serde_json::to_string(query)?;
        writeln!(writer, "{}", line)?;
    }

    writer.flush()?;
    Ok(())
}

/// Write relevance judgments to a BEIR qrels.tsv file
pub fn write_qrels(path: &Path, qrels: &[Qrel]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let file = File::create(path)
        .with_context(|| format!("Failed to create qrels file: {}", path.display()))?;
    let mut writer = BufWriter::new(file);

    // Write header
    writeln!(writer, "query-id\tdoc-id\tscore")?;

    for qrel in qrels {
        writeln!(writer, "{}\t{}\t{}", qrel.query_id, qrel.doc_id, qrel.score)?;
    }

    writer.flush()?;
    Ok(())
}

/// Read documents from a BEIR corpus.jsonl file
pub fn read_corpus(path: &Path) -> Result<Vec<BeirDocument>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open corpus file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut documents = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let doc: BeirDocument = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse document at line {}", line_num + 1))?;
        documents.push(doc);
    }

    Ok(documents)
}

/// Read queries from a BEIR queries.jsonl file
pub fn read_queries(path: &Path) -> Result<Vec<BeirQuery>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open queries file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut queries = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let query: BeirQuery = serde_json::from_str(&line)
            .with_context(|| format!("Failed to parse query at line {}", line_num + 1))?;
        queries.push(query);
    }

    Ok(queries)
}

/// Read qrels from a BEIR qrels.tsv file
pub fn read_qrels(path: &Path) -> Result<Vec<Qrel>> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open qrels file: {}", path.display()))?;
    let reader = BufReader::new(file);

    let mut qrels = Vec::new();
    for (line_num, line) in reader.lines().enumerate() {
        let line = line?;
        // Skip header
        if line_num == 0 && line.starts_with("query-id") {
            continue;
        }
        if line.trim().is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            anyhow::bail!("Invalid qrel format at line {}", line_num + 1);
        }

        qrels.push(Qrel {
            query_id: parts[0].to_string(),
            doc_id: parts[1].to_string(),
            score: parts[2].parse()?,
        });
    }

    Ok(qrels)
}

/// Append a single document to corpus file
pub fn append_document(path: &Path, doc: &BeirDocument) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open corpus file for append: {}", path.display()))?;

    let line = serde_json::to_string(doc)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

/// Append a single query to queries file
pub fn append_query(path: &Path, query: &BeirQuery) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("Failed to open queries file for append: {}", path.display()))?;

    let line = serde_json::to_string(query)?;
    writeln!(file, "{}", line)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_corpus_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("corpus.jsonl");

        let docs = vec![
            BeirDocument {
                id: "doc1".to_string(),
                title: "Title 1".to_string(),
                text: "Text content 1".to_string(),
            },
            BeirDocument {
                id: "doc2".to_string(),
                title: "Title 2".to_string(),
                text: "Text content 2".to_string(),
            },
        ];

        write_corpus(&path, &docs).unwrap();
        let loaded = read_corpus(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].id, "doc1");
        assert_eq!(loaded[1].text, "Text content 2");
    }

    #[test]
    fn test_qrels_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("qrels.tsv");

        let qrels = vec![
            Qrel {
                query_id: "q1".to_string(),
                doc_id: "doc1".to_string(),
                score: 3,
            },
            Qrel {
                query_id: "q1".to_string(),
                doc_id: "doc2".to_string(),
                score: 1,
            },
        ];

        write_qrels(&path, &qrels).unwrap();
        let loaded = read_qrels(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].score, 3);
    }
}
