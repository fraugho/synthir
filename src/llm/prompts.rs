use crate::topics::TopicConfig;

/// System prompt for document generation
pub fn document_system_prompt(topic: &str) -> String {
    format!(
        r#"You are a document generator for information retrieval datasets.
Generate realistic {} documents that users would actually have.
Output ONLY the document content, no explanations or metadata."#,
        topic
    )
}

/// User prompt for document generation
pub fn document_user_prompt(topic_config: &TopicConfig, index: usize) -> String {
    format!(
        r#"Generate a realistic {} document.

Requirements:
- Length: {}-{} words
- Style: {}
- Include realistic details a real document would have
{}

This is document #{} in the collection, so make it unique and different from others.

Output ONLY the document text, nothing else."#,
        topic_config.name,
        topic_config.min_words,
        topic_config.max_words,
        topic_config.style_description,
        topic_config.specific_instructions,
        index + 1
    )
}

/// Prompt for natural language query generation (casual Google-style)
pub fn natural_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate REALISTIC Google search questions that regular people type.

Document:
"""
{}
"""

Generate a casual question a normal person would Google:
- Simple, everyday language (NOT academic or formal)
- Like texting a friend or asking Google
- May be grammatically imperfect or lowercase
- Short and to the point (under 12 words usually)
- Examples of REAL natural searches:
  - "how do i fix a leaky faucet"
  - "is it bad to eat expired yogurt"
  - "why does my dog keep scratching"
  - "whats the best way to learn python"
  - "can you freeze cooked pasta"
  - "how long to boil eggs"

The question should be answerable by the document.

Output ONLY the question, nothing else."#,
        document_text
    )
}

/// Prompt for keyword query generation (realistic messy user searches)
pub fn keyword_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate REALISTIC search queries that mimic how real users type.

Document:
"""
{}
"""

Generate a keyword search like a REAL person would type into Google:
- 1-4 words, usually incomplete fragments
- May have typos or misspellings (about 15% of real searches do)
- No perfect grammar - just quick search terms
- Often missing words or using shorthand
- Examples of REAL keyword searches:
  - "pasta recipe easy"
  - "why wont car start"
  - "best laptop 2024"
  - "headache wont go away"
  - "recipie chicken" (typo)
  - "python tutorial beginer" (typo)
  - "how to"
  - "iphone not charging"

Output ONLY the search terms, nothing else. No quotes."#,
        document_text
    )
}

/// Prompt for complex/multi-hop query generation
pub fn complex_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate search queries for information retrieval datasets.

Document:
"""
{}
"""

Generate a complex query that requires reasoning to connect to this document.
Examples of complex queries:
- Comparative: "difference between X and Y"
- Multi-step: "how to do X after Y"
- Conditional: "best approach for X when Y"

The document should be relevant to answering this query, but not trivially so.

Output ONLY the query, nothing else. No quotes, no explanation."#,
        document_text
    )
}

/// Prompt for academic/formal query generation
pub fn academic_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate detailed, academic-style queries for information retrieval.

Document:
"""
{}
"""

Generate a formal, detailed query that a researcher or expert might use:
- Specific terminology and precise language
- Well-structured question or information need
- May reference specific concepts, methods, or metrics
- Longer and more detailed than typical user searches
- Examples of academic queries:
  - "What is the correlation between urban green space coverage and PM2.5 reduction in metropolitan areas?"
  - "How does the Maillard reaction temperature affect flavor compound formation in bread crusts?"
  - "What are the primary mechanisms by which SSRIs modulate serotonin reuptake in synaptic clefts?"
  - "How do transformer attention mechanisms compare to LSTM gates for sequence modeling?"

The document should contain information relevant to this query.

Output ONLY the query, nothing else."#,
        document_text
    )
}

/// Prompt for semantic query generation (tests embedding-based retrieval)
pub fn semantic_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate SEMANTIC search queries that test embedding-based retrieval.

Document:
"""
{}
"""

Generate a query that captures the document's meaning WITHOUT using the same keywords:
- Use SYNONYMS instead of exact terms (cats -> felines/kittens, car -> automobile/vehicle)
- PARAPHRASE concepts - do not copy phrases from the document
- Focus on MEANING, not specific words
- A BM25/keyword search should FAIL to match this query
- An embedding search should SUCCEED

Examples of semantic rewording:
- Doc about "cooking pasta" -> Query: "preparing Italian noodles"
- Doc about "fixing leaky faucet" -> Query: "repairing dripping tap"
- Doc about "machine learning models" -> Query: "training AI systems"

The query MUST be relevant but use DIFFERENT vocabulary.

Output ONLY the query, nothing else."#,
        document_text
    )
}

/// Prompt for relevance scoring
pub fn relevance_scoring_prompt(query: &str, document_text: &str) -> String {
    format!(
        r#"You are a relevance judge for information retrieval.

Query: "{}"

Document:
"""
{}
"""

Rate the document's relevance to the query on a 0-3 scale:
0 = Not relevant (document does not help answer the query)
1 = Marginally relevant (mentions related concepts but doesn't answer)
2 = Relevant (partially answers or is useful for the query)
3 = Highly relevant (directly and completely answers the query)

Output ONLY a single digit (0, 1, 2, or 3). Nothing else."#,
        query, document_text
    )
}

/// Prompt for hard negative validation
pub fn hard_negative_validation_prompt(query: &str, document_text: &str) -> String {
    format!(
        r#"Query: "{}"

Document:
"""
{}
"""

Is this document relevant to the query? Answer only "yes" or "no"."#,
        query, document_text
    )
}

/// Prompt for fine-grained relevance scoring (0-100 scale)
/// Used in pooled scoring mode for cliff detection
pub fn fine_grained_relevance_prompt(query: &str, document_text: &str) -> String {
    format!(
        r#"You are a relevance judge for information retrieval.

Query: "{}"

Document:
"""
{}
"""

Rate how relevant this document is to the query on a scale of 0-100:
- 0-10: Completely irrelevant, no connection to the query
- 11-30: Barely relevant, mentions tangentially related concepts
- 31-50: Somewhat relevant, discusses related topics but doesn't answer
- 51-70: Relevant, provides useful information for the query
- 71-90: Highly relevant, directly addresses most of the query
- 91-100: Perfectly relevant, completely and directly answers the query

Output ONLY a number from 0 to 100. Nothing else."#,
        query, document_text
    )
}

/// Prompt for custom range relevance scoring
/// Allows user-defined min/max score range
pub fn range_relevance_prompt(query: &str, document_text: &str, min_score: u16, max_score: u16) -> String {
    let range = max_score - min_score;
    let low_threshold = min_score + range / 4;
    let mid_threshold = min_score + range / 2;
    let high_threshold = min_score + (range * 3) / 4;

    format!(
        r#"You are a relevance judge for information retrieval.

Query: "{}"

Document:
"""
{}
"""

Rate how relevant this document is to the query on a scale of {}-{}:
- {}-{}: Not relevant (document does not help answer the query)
- {}-{}: Marginally relevant (mentions related concepts but doesn't answer)
- {}-{}: Relevant (partially answers or is useful for the query)
- {}-{}: Highly relevant (directly and completely answers the query)

Output ONLY a number from {} to {}. Nothing else."#,
        query, document_text,
        min_score, max_score,
        min_score, low_threshold,
        low_threshold + 1, mid_threshold,
        mid_threshold + 1, high_threshold,
        high_threshold + 1, max_score,
        min_score, max_score
    )
}

/// Prompt for LLM-generated topic
pub fn topic_generation_prompt() -> &'static str {
    r#"Generate a random, specific topic for a document collection.
NOT generic topics like "technology" or "science".
Be specific and interesting, like:
- "vintage motorcycle restoration guides"
- "sourdough bread troubleshooting"
- "apartment lease agreements"
- "D&D campaign session notes"

Output ONLY the topic name, 2-5 words, nothing else."#
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prompts_not_empty() {
        assert!(!document_system_prompt("recipes").is_empty());
        assert!(!natural_query_prompt("test doc").is_empty());
        assert!(!keyword_query_prompt("test doc").is_empty());
        assert!(!academic_query_prompt("test doc").is_empty());
        assert!(!complex_query_prompt("test doc").is_empty());
        assert!(!semantic_query_prompt("test doc").is_empty());
        assert!(!relevance_scoring_prompt("query", "doc").is_empty());
        assert!(!hard_negative_validation_prompt("query", "doc").is_empty());
        assert!(!topic_generation_prompt().is_empty());
    }
}
