use crate::config::{GenerationConfig, QueryType, RuntimeOptions};
use crate::output::OutputOrganizer;
use crate::topics::TopicConfig;

/// Display dry-run information
pub fn display_dry_run_info(
    config: &GenerationConfig,
    topic_config: &TopicConfig,
    _options: &RuntimeOptions,
) {
    println!("\n=== DRY RUN MODE ===\n");
    println!("This shows what would be generated without making actual LLM calls.\n");

    println!("Configuration:");
    println!("  Topic: {} ({})", topic_config.name, topic_config.description);
    println!("  Documents: {}", config.document_count);
    println!("  Queries per type: {}", config.queries_per_type);
    println!("  Output directory: {}", config.output_dir.display());
    println!("  Model: {}", config.model);
    println!("  Base URL: {}", config.base_url);
    println!();

    println!("Document Generation:");
    println!("  Word range: {}-{} words", topic_config.min_words, topic_config.max_words);
    println!("  Style: {}", topic_config.style_description);
    println!();

    println!("Query Generation:");
    for query_type in QueryType::all_types() {
        println!("  {} queries: {}", query_type, config.queries_per_type);
    }
    println!("  mixed queries: {}", config.queries_per_type);
    println!();

    println!("Output Structure:");
    let organizer = OutputOrganizer::new(config.output_dir.clone(), topic_config.name.clone());
    for path in organizer.all_paths() {
        println!("  {}", path.display());
    }
    println!();

    println!("Estimated API Calls:");
    let doc_calls = config.document_count;
    let query_calls = config.queries_per_type * 4; // 3 types + mixed
    let relevance_calls = query_calls; // At minimum, source pairs
    let hard_neg_calls = if config.generate_hard_negatives {
        query_calls // Rough estimate
    } else {
        0
    };
    let total = doc_calls + query_calls + relevance_calls + hard_neg_calls;

    println!("  Document generation: ~{}", doc_calls);
    println!("  Query generation: ~{}", query_calls);
    println!("  Relevance scoring: ~{}", relevance_calls);
    if config.generate_hard_negatives {
        println!("  Hard negative mining: ~{}", hard_neg_calls);
    }
    println!("  Total: ~{}", total);
    println!();

    println!("=== END DRY RUN ===\n");
}

/// Display verbose generation progress
pub fn display_verbose_start(config: &GenerationConfig, topic_config: &TopicConfig) {
    println!("\n[VERBOSE] Starting dataset generation");
    println!("[VERBOSE] Topic: {}", topic_config.name);
    println!("[VERBOSE] Documents: {}", config.document_count);
    println!("[VERBOSE] Queries per type: {}", config.queries_per_type);
    println!("[VERBOSE] Output: {}", config.output_dir.display());
    println!();
}

/// Display available topics
pub fn display_topics(topics: &[(String, String)]) {
    println!("\nAvailable Topics:\n");
    println!("{:<20} {}", "NAME", "DESCRIPTION");
    println!("{:<20} {}", "----", "-----------");
    for (name, desc) in topics {
        println!("{:<20} {}", name, desc);
    }
    println!();
    println!("Use --topic <name> to select a topic.");
    println!("Use --topic llm-generated to let the LLM pick a random topic.");
    println!();
}

/// Display resume info
pub fn display_resume_info(
    completed_docs: usize,
    total_docs: usize,
    completed_queries: &[(QueryType, usize)],
    phase: &str,
) {
    println!("\n[RESUME] Resuming from checkpoint");
    println!("[RESUME] Current phase: {}", phase);
    println!("[RESUME] Documents: {}/{} completed", completed_docs, total_docs);
    for (query_type, count) in completed_queries {
        println!("[RESUME] {} queries: {} completed", query_type, count);
    }
    println!();
}

/// Format a duration nicely
pub fn format_duration(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m {}s", secs / 60, secs % 60)
    } else {
        format!("{}h {}m {}s", secs / 3600, (secs % 3600) / 60, secs % 60)
    }
}
