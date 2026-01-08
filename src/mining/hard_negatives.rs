use crate::llm::{hard_negative_validation_prompt, LLMProvider};
use crate::output::{BeirDocument, BeirQuery, Qrel};
use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};
use std::collections::{HashMap, HashSet};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Schema, TEXT, STORED, STRING, Value};
use tantivy::{doc, Index, IndexWriter, ReloadPolicy, TantivyDocument};
use tracing::info;

/// Hard negative miner using BM25 + LLM validation
pub struct HardNegativeMiner<'a> {
    provider: &'a LLMProvider,
    dry_run: bool,
}

impl<'a> HardNegativeMiner<'a> {
    pub fn new(provider: &'a LLMProvider, dry_run: bool) -> Self {
        Self { provider, dry_run }
    }

    /// Build a BM25 index from documents
    fn build_index(&self, documents: &[BeirDocument]) -> Result<(Index, Schema)> {
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

    /// Find candidate hard negatives using BM25
    fn find_bm25_candidates(
        &self,
        query: &BeirQuery,
        index: &Index,
        schema: &Schema,
        top_k: usize,
        exclude_doc_ids: &HashSet<String>,
    ) -> Result<Vec<String>> {
        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;
        let searcher = reader.searcher();

        let text_field = schema.get_field("text").unwrap();
        let doc_id_field = schema.get_field("doc_id").unwrap();

        let query_parser = QueryParser::for_index(index, vec![text_field]);

        // Parse query, escaping special characters
        let escaped_query = escape_query(&query.text);
        let parsed_query = match query_parser.parse_query(&escaped_query) {
            Ok(q) => q,
            Err(_) => {
                // Fallback to simple term query if parsing fails
                let terms: Vec<&str> = query.text.split_whitespace().take(3).collect();
                query_parser.parse_query(&terms.join(" "))?
            }
        };

        let top_docs = searcher.search(&parsed_query, &TopDocs::with_limit(top_k * 2))?;

        let mut candidates = Vec::new();
        for (_score, doc_address) in top_docs {
            let retrieved_doc: TantivyDocument = searcher.doc(doc_address)?;
            if let Some(doc_id) = retrieved_doc.get_first(doc_id_field) {
                if let Some(doc_id_str) = doc_id.as_str() {
                    if !exclude_doc_ids.contains::<str>(doc_id_str) {
                        candidates.push(doc_id_str.to_string());
                        if candidates.len() >= top_k {
                            break;
                        }
                    }
                }
            }
        }

        Ok(candidates)
    }

    /// Validate that a candidate is actually a hard negative (not relevant)
    async fn validate_hard_negative(
        &self,
        query: &BeirQuery,
        doc: &BeirDocument,
    ) -> Result<bool> {
        if self.dry_run {
            // In dry run, assume it's a valid hard negative
            return Ok(true);
        }

        let prompt = hard_negative_validation_prompt(&query.text, &doc.text);
        let is_relevant = self.provider.validate_yes_no(&prompt).await?;

        // It's a valid hard negative if the LLM says it's NOT relevant
        Ok(!is_relevant)
    }

    /// Mine hard negatives for all queries
    pub async fn mine_hard_negatives(
        &self,
        queries: &[BeirQuery],
        documents: &[BeirDocument],
        existing_qrels: &[Qrel],
        hard_negatives_per_query: usize,
    ) -> Result<Vec<Qrel>> {
        info!("Building BM25 index for {} documents...", documents.len());
        let (index, schema) = self.build_index(documents)?;

        // Build document lookup
        let doc_map: HashMap<&str, &BeirDocument> =
            documents.iter().map(|d| (d.id.as_str(), d)).collect();

        // Build set of relevant (query_id -> doc_ids) from existing qrels
        let mut relevant_docs: HashMap<String, HashSet<String>> = HashMap::new();
        for qrel in existing_qrels {
            if qrel.score > 0 {
                relevant_docs
                    .entry(qrel.query_id.clone())
                    .or_default()
                    .insert(qrel.doc_id.clone());
            }
        }

        let mut hard_negatives = Vec::new();

        let pb = ProgressBar::new(queries.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} queries for hard negatives ({eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        for query in queries {
            let exclude = relevant_docs.get(&query.id).cloned().unwrap_or_default();

            // Find BM25 candidates
            let candidates = self.find_bm25_candidates(
                query,
                &index,
                &schema,
                hard_negatives_per_query * 2, // Get extra to account for validation failures
                &exclude,
            )?;

            let mut found = 0;
            for doc_id in candidates {
                if found >= hard_negatives_per_query {
                    break;
                }

                if let Some(doc) = doc_map.get(doc_id.as_str()) {
                    let is_hard_negative = self.validate_hard_negative(query, doc).await?;

                    if is_hard_negative {
                        hard_negatives.push(Qrel {
                            query_id: query.id.clone(),
                            doc_id: doc_id.clone(),
                            score: 0, // Hard negative = score 0
                        });
                        found += 1;
                    }
                }
            }

            pb.inc(1);
        }

        pb.finish_with_message("Hard negative mining complete");
        info!("Mined {} hard negatives", hard_negatives.len());

        Ok(hard_negatives)
    }
}

/// Escape special characters for tantivy query parser
fn escape_query(query: &str) -> String {
    let special_chars = ['+', '-', '&', '|', '!', '(', ')', '{', '}', '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/'];
    let mut escaped = String::with_capacity(query.len() * 2);

    for c in query.chars() {
        if special_chars.contains(&c) {
            escaped.push('\\');
        }
        escaped.push(c);
    }

    escaped
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_escape_query() {
        assert_eq!(escape_query("hello world"), "hello world");
        assert_eq!(escape_query("what is 2+2?"), "what is 2\\+2\\?");
        assert_eq!(escape_query("test (foo)"), "test \\(foo\\)");
    }
}
