use serde::{Deserialize, Serialize};

/// Configuration for a document topic
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopicConfig {
    /// Topic identifier (e.g., "recipes", "miscellaneous")
    pub name: String,

    /// Human-readable description
    pub description: String,

    /// Minimum word count for documents
    pub min_words: usize,

    /// Maximum word count for documents
    pub max_words: usize,

    /// Style description for the LLM
    pub style_description: String,

    /// Topic-specific instructions for document generation
    pub specific_instructions: String,
}

impl TopicConfig {
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        min_words: usize,
        max_words: usize,
        style_description: impl Into<String>,
        specific_instructions: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            min_words,
            max_words,
            style_description: style_description.into(),
            specific_instructions: specific_instructions.into(),
        }
    }
}
