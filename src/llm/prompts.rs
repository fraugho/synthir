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

/// Prompt for two-phase document generation: first generate a title/concept
pub fn document_idea_prompt(topic_config: &TopicConfig, existing_ideas: &[String], category: Option<&str>) -> String {
    let exclusion = if existing_ideas.is_empty() {
        String::new()
    } else {
        let ideas_list = existing_ideas.iter()
            .take(50) // Limit context size
            .map(|s| format!("- {}", s))
            .collect::<Vec<_>>()
            .join("\n");
        format!(
            r#"

IMPORTANT: Do NOT generate any of these already-used ideas:
{}

Your idea must be DIFFERENT from all of the above."#,
            ideas_list
        )
    };

    let category_instruction = category.map(|c| format!(
        "\n\nCATEGORY CONSTRAINT: The document MUST be about: {}", c
    )).unwrap_or_default();

    format!(
        r#"Generate a unique title/concept for a {} document.

Requirements:
- Style: {}
- Be specific and creative
- This will become a full document later{}{}

Output ONLY a short title or concept (5-15 words), nothing else. /no_think"#,
        topic_config.name,
        topic_config.style_description,
        category_instruction,
        exclusion
    )
}

/// Prompt to expand a document idea into a full document
pub fn document_expand_prompt(topic_config: &TopicConfig, idea: &str) -> String {
    format!(
        r#"Expand this concept into a full {} document:

Concept: "{}"

Requirements:
- Length: {}-{} words
- Style: {}
- Include realistic details a real document would have
{}

Output ONLY the document content, nothing else."#,
        topic_config.name,
        idea,
        topic_config.min_words,
        topic_config.max_words,
        topic_config.style_description,
        topic_config.specific_instructions
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

Output ONLY the question, nothing else. /no_think"#,
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
- 2-6 words, usually incomplete fragments
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

Output ONLY the search terms, nothing else. No quotes. /no_think"#,
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

Output ONLY the query, nothing else. No quotes, no explanation. /no_think"#,
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

Output ONLY the query, nothing else. /no_think"#,
        document_text
    )
}

/// Prompt for semantic query generation (tests embedding-based retrieval)
pub fn semantic_query_prompt(document_text: &str) -> String {
    format!(
        r#"Generate a search query with ABSOLUTELY ZERO word overlap with the document.

Document:
"""
{}
"""

TASK: Create a query to find this document using ONLY synonyms and category terms.

STRICT RULES - YOUR QUERY WILL BE AUTOMATICALLY REJECTED IF:
- ANY word from your query appears in the document (even partial matches)
- You use morphological variants (enzyme/enzymes, test/testing, cell/cells)
- You use any proper nouns, names, or specific terms from the document

WHAT TO DO:
1. Identify the TOPIC/CATEGORY of the document (not specific terms)
2. Use COMPLETELY DIFFERENT words that mean similar things
3. Think: "How would I describe this to someone without using any words from it?"

GOOD EXAMPLES (zero overlap):
- Doc about "enzyme activity temperature experiment" -> "biological catalyst thermal behavior"
- Doc about "cell membrane transport" -> "biological barrier movement mechanisms"
- Doc about "Berlin restaurant Italian food" -> "European dining establishment"
- Doc about "iPhone battery not charging" -> "mobile device power issues"
- Doc about "Python machine learning tutorial" -> "programming AI educational guide"

BAD EXAMPLES (has overlap - REJECTED):
- Doc about "enzyme activity" -> "enzyme testing" (WRONG: "enzyme" appears in doc)
- Doc about "Berlin restaurant" -> "Berlin food" (WRONG: "Berlin" appears in doc)
- Doc about "cell biology" -> "cell processes" (WRONG: "cell" appears in doc)

Your query must be 2-6 words using ONLY words that DO NOT appear in the document above.

Output ONLY the query, nothing else. /no_think"#,
        document_text
    )
}

/// Prompt for basic query generation (partial keyword matching)
pub fn basic_query_prompt(document_text: &str) -> String {
    format!(
        r#"You generate document-finding queries with PARTIAL keyword matching.

Document:
"""
{}
"""

Generate a query someone would use to FIND this document, with SOME but NOT ALL keywords matching:
- This is a DOCUMENT SEARCH - you're looking for this document
- Include 1-2 words that appear in the document
- OMIT other key identifying words to make it a partial match
- BM25/lexical search should find this, but not as the top result
- Query should be incomplete or abbreviated

RULES:
- Read the FULL document content above
- Pick the most descriptive 2-4 words, dropping some key terms
- Keep it like a quick, lazy search someone might type

Examples (showing partial keyword retention):
- Doc: "Encyclopedia of Emperor Penguins" -> Query: "penguin book" (dropped "emperor", "encyclopedia")
- Doc: "Toyota Camry 2019 Owner's Manual" -> Query: "camry manual" (dropped "toyota", "2019", "owner")
- Doc: "Introduction to Machine Learning with Python" -> Query: "machine learning python" (dropped "introduction")
- Doc: "The Complete Guide to Mediterranean Cooking" -> Query: "mediterranean recipes" (dropped "complete", "guide", changed "cooking")
- Doc: "Advanced Cardiovascular Life Support Manual" -> Query: "cardiac life support" (dropped "advanced", changed "cardiovascular")
- Doc about fixing iPhone battery issues -> Query: "iphone battery" (partial match)

The query should be 2-4 words with partial overlap.

Output ONLY the query, nothing else. /no_think"#,
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

Output ONLY a single digit (0, 1, 2, or 3). Nothing else. /no_think"#,
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

Is this document relevant to the query? Answer only "yes" or "no". /no_think"#,
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

Output ONLY a number from 0 to 100. Nothing else. /no_think"#,
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

Output ONLY a number from {} to {}. Nothing else. /no_think"#,
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

Output ONLY the topic name, 2-5 words, nothing else. /no_think"#
}

/// Prompt for language detection
/// Takes multiple text samples and asks LLM to identify the language
pub fn language_detection_prompt(samples: &[&str]) -> String {
    let samples_text = samples
        .iter()
        .enumerate()
        .map(|(i, s)| format!("Sample {}:\n\"\"\"\n{}\n\"\"\"", i + 1, s))
        .collect::<Vec<_>>()
        .join("\n\n");

    format!(
        r#"Identify the language of these text samples.

{}

What language are these samples written in?

IMPORTANT:
- Look at ALL samples and determine the PRIMARY language
- If samples are mixed languages, identify the MAJORITY language
- Use the standard English name for the language (e.g., "French", "German", "Japanese", "English")

Output ONLY the language name, nothing else. Single word like: English, French, German, Japanese, Spanish, Italian, Arabic, Chinese, Korean, Portuguese, Russian, etc. /no_think"#,
        samples_text
    )
}

/// Add language instruction to a query prompt if language is specified
pub fn with_language_instruction(prompt: String, language: Option<&str>) -> String {
    match language {
        Some(lang) if lang.to_lowercase() != "english" => {
            format!(
                "{}\n\nIMPORTANT: Generate the query in {}. Do NOT translate to English.",
                prompt, lang
            )
        }
        _ => prompt,
    }
}

/// Prompt for translating a query to a target language
pub fn translation_prompt(query: &str, target_language: &str) -> String {
    format!(
        r#"Translate this search query to {}.

Query: "{}"

RULES:
- Translate naturally, not word-for-word
- Keep the same search intent and meaning
- Use natural phrasing a {} speaker would use
- If the query contains proper nouns, keep them as-is or use the common {} form

Output ONLY the translated query, nothing else. /no_think"#,
        target_language, query, target_language, target_language
    )
}

/// Default languages for translation mode
pub const DEFAULT_TRANSLATE_LANGUAGES: &[&str] = &[
    "Spanish", "French", "German", "Japanese", "Chinese",
    "Arabic", "Korean", "Portuguese", "Russian", "Italian",
    "Dutch", "Polish", "Turkish", "Vietnamese", "Thai",
];

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
        assert!(!basic_query_prompt("test doc").is_empty());
        assert!(!relevance_scoring_prompt("query", "doc").is_empty());
        assert!(!hard_negative_validation_prompt("query", "doc").is_empty());
        assert!(!topic_generation_prompt().is_empty());
    }
}
