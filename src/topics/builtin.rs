use super::TopicConfig;
use std::collections::HashMap;

/// Get all built-in topics
pub fn get_builtin_topics() -> HashMap<String, TopicConfig> {
    let mut topics = HashMap::new();

    topics.insert(
        "miscellaneous".to_string(),
        TopicConfig::new(
            "miscellaneous",
            "Random folder chaos - the hard dataset with no theme",
            50,
            500,
            "Completely varied - could be anything from any context",
            r#"This could be ANYTHING - a receipt, a poem, meeting notes, a shopping list,
a diary entry, code snippet, random webpage saved, bookmark description,
screenshot OCR text, chat log excerpt, to-do list, warranty info, etc.
Be unpredictable and diverse. Each document should feel like it came from
a completely different context. Include realistic artifacts like dates,
names, partial information, abbreviations."#,
        ),
    );

    topics.insert(
        "recipes".to_string(),
        TopicConfig::new(
            "recipes",
            "Cooking recipes with ingredients and instructions",
            100,
            400,
            "Recipe format with title, ingredients, and steps",
            r#"Include title, ingredients list with measurements, and numbered steps.
May have casual notes like "my grandma's version" or "adapted from X".
Include cooking times, serving sizes, and occasional tips or variations.
Some may have brief personal stories or origin notes."#,
        ),
    );

    topics.insert(
        "tiny-notes".to_string(),
        TopicConfig::new(
            "tiny-notes",
            "Very short notes and snippets - tests IR on short documents",
            10,
            50,
            "Brief, like a sticky note or quick reminder",
            r#"Very brief, 10-50 words max. Like a sticky note or quick reminder.
May have typos, abbreviations, incomplete sentences.
Examples: "Call dentist tmrw 3pm", "milk eggs bread", "meeting notes: discuss Q4 budget",
"password hint: pet name + birth year", "return amazon pkg by friday"."#,
        ),
    );

    topics.insert(
        "technical-docs".to_string(),
        TopicConfig::new(
            "technical-docs",
            "Code documentation, READMEs, API references",
            150,
            800,
            "Technical documentation style with code examples",
            r#"Technical jargon, code blocks, API descriptions.
Include function signatures, parameter descriptions, return values.
May have installation instructions, usage examples, troubleshooting sections.
Use markdown formatting where appropriate."#,
        ),
    );

    topics.insert(
        "legal".to_string(),
        TopicConfig::new(
            "legal",
            "Contracts, terms of service, policies",
            200,
            1000,
            "Dense formal legal language",
            r#"Dense formal language with legal terminology.
Include sections, subsections, definitions, obligations.
May reference specific laws, jurisdictions, parties.
Use formal structure with numbered clauses."#,
        ),
    );

    topics.insert(
        "emails".to_string(),
        TopicConfig::new(
            "emails",
            "Email threads and messages",
            50,
            300,
            "Conversational email format",
            r#"Include To/From/Subject headers. May be part of a thread.
Conversational tone, may have signatures, disclaimers.
Include context references ("as discussed", "per our call").
May have attachments mentioned, CC/BCC references."#,
        ),
    );

    topics.insert(
        "academic".to_string(),
        TopicConfig::new(
            "academic",
            "Research papers, articles, academic writing",
            300,
            1200,
            "Formal academic writing with citations",
            r#"Formal academic style with citations in brackets [1].
Include abstract-like summaries, methodology hints, findings.
Use discipline-specific terminology.
May reference other works, include data/statistics."#,
        ),
    );

    topics.insert(
        "product-reviews".to_string(),
        TopicConfig::new(
            "product-reviews",
            "User product reviews and ratings",
            30,
            250,
            "Consumer review style, opinionated",
            r#"Opinionated, varied quality from detailed to brief.
Include pros/cons, use cases, comparisons.
May have star ratings mentioned, purchase context.
Range from enthusiastic to disappointed tones."#,
        ),
    );

    topics
}

/// Get a specific built-in topic by name
pub fn get_topic(name: &str) -> Option<TopicConfig> {
    get_builtin_topics().remove(name)
}

/// List all available topic names
pub fn list_topic_names() -> Vec<String> {
    get_builtin_topics().keys().cloned().collect()
}

/// Create a topic config from an LLM-generated topic name
pub fn create_custom_topic(name: String, description: String) -> TopicConfig {
    TopicConfig::new(
        name.clone(),
        description,
        100,
        500,
        "Natural, realistic style appropriate for the topic",
        format!(
            r#"Generate realistic {} documents.
Include appropriate formatting, terminology, and structure for this topic.
Make each document feel authentic and varied."#,
            name
        ),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_topics_exist() {
        let topics = get_builtin_topics();
        assert!(topics.contains_key("miscellaneous"));
        assert!(topics.contains_key("recipes"));
        assert!(topics.contains_key("tiny-notes"));
        assert!(topics.contains_key("technical-docs"));
        assert!(topics.contains_key("legal"));
        assert!(topics.contains_key("emails"));
        assert!(topics.contains_key("academic"));
        assert!(topics.contains_key("product-reviews"));
    }

    #[test]
    fn test_get_topic() {
        let topic = get_topic("recipes");
        assert!(topic.is_some());
        let topic = topic.unwrap();
        assert_eq!(topic.name, "recipes");
    }
}
