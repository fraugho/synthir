use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::chat::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessage,
        ChatCompletionRequestSystemMessageContent, ChatCompletionRequestUserMessage,
        ChatCompletionRequestUserMessageContent, CreateChatCompletionRequest,
    },
    Client,
};
use std::sync::atomic::{AtomicUsize, Ordering};
use thiserror::Error;
use tracing::{debug, info, warn};

/// Configuration for the LLM provider
#[derive(Debug, Clone)]
pub struct LLMProviderConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub max_retries: u32,
}

impl Default for LLMProviderConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-4".to_string(),
            max_retries: 3,
        }
    }
}

/// Errors that can occur during LLM operations
#[derive(Error, Debug)]
pub enum LLMError {
    #[error("Failed to parse LLM output: {0}")]
    ParseError(String),

    #[error("Invalid score output: expected 0-3, got '{0}'")]
    InvalidScore(String),

    #[error("Invalid fine score output: expected 0-100, got '{0}'")]
    InvalidFineScore(String),

    #[error("Invalid range score output: expected {1}-{2}, got '{0}'")]
    InvalidRangeScore(String, u16, u16),

    #[error("Invalid yes/no output: expected 'yes' or 'no', got '{0}'")]
    InvalidYesNo(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Max retries exceeded")]
    MaxRetriesExceeded,
}

/// Type of output expected from the LLM
#[derive(Debug, Clone, Copy)]
pub enum OutputType {
    Score,
    FineScore, // 0-100 scale
    Query,
    Document,
    YesNo,
    Topic,
}

/// Range bounds for custom range scoring
#[derive(Debug, Clone, Copy)]
pub struct ScoreRange {
    pub min: u16,
    pub max: u16,
}

/// LLM provider that wraps async-openai for OpenAI-compatible APIs
pub struct LLMProvider {
    client: Client<OpenAIConfig>,
    model: String,
    max_retries: u32,
}

impl LLMProvider {
    /// Create a new LLM provider with the given configuration
    pub fn new(config: LLMProviderConfig) -> Result<Self> {
        let openai_config = OpenAIConfig::new()
            .with_api_key(&config.api_key)
            .with_api_base(&config.base_url);

        let client = Client::with_config(openai_config);

        Ok(Self {
            client,
            model: config.model,
            max_retries: config.max_retries,
        })
    }

    /// Send a completion request to the LLM
    pub async fn complete(&self, system_prompt: Option<&str>, user_prompt: &str) -> Result<String> {
        let mut messages: Vec<ChatCompletionRequestMessage> = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(ChatCompletionRequestMessage::System(
                ChatCompletionRequestSystemMessage {
                    content: ChatCompletionRequestSystemMessageContent::Text(sys.to_string()),
                    name: None,
                },
            ));
        }

        messages.push(ChatCompletionRequestMessage::User(
            ChatCompletionRequestUserMessage {
                content: ChatCompletionRequestUserMessageContent::Text(user_prompt.to_string()),
                name: None,
            },
        ));

        let request = CreateChatCompletionRequest {
            model: self.model.clone(),
            messages,
            temperature: Some(0.7),
            ..Default::default()
        };

        let response = self
            .client
            .chat()
            .create(request)
            .await
            .context("Failed to get LLM response")?;

        let content = response
            .choices
            .first()
            .and_then(|c| c.message.content.clone())
            .unwrap_or_default();

        debug!("LLM response: {}", content);

        Ok(content)
    }

    /// Parse LLM output based on expected type
    pub fn parse_output(raw: &str, expected: OutputType) -> Result<String, LLMError> {
        let cleaned = raw.trim();

        match expected {
            OutputType::Score => {
                // Extract first digit 0-3
                let score = cleaned
                    .chars()
                    .find(|c| c.is_ascii_digit())
                    .and_then(|c| c.to_digit(10))
                    .filter(|&d| d <= 3)
                    .ok_or_else(|| LLMError::InvalidScore(cleaned.to_string()))?;
                Ok(score.to_string())
            }
            OutputType::FineScore => {
                // Extract number 0-100
                let num_str: String = cleaned
                    .chars()
                    .filter(|c| c.is_ascii_digit())
                    .take(3) // Max 3 digits for 100
                    .collect();
                let score: u8 = num_str
                    .parse()
                    .map_err(|_| LLMError::InvalidFineScore(cleaned.to_string()))?;
                if score > 100 {
                    return Err(LLMError::InvalidFineScore(cleaned.to_string()));
                }
                Ok(score.to_string())
            }
            OutputType::Query | OutputType::Topic => {
                // Remove quotes if present, take first line
                let query = cleaned
                    .trim_matches('"')
                    .trim_matches('\'')
                    .lines()
                    .next()
                    .unwrap_or(cleaned)
                    .trim();
                Ok(query.to_string())
            }
            OutputType::Document => {
                // Take full content, remove any "Here is..." preamble
                let doc = if cleaned.to_lowercase().starts_with("here") {
                    cleaned.splitn(2, '\n').nth(1).unwrap_or(cleaned)
                } else {
                    cleaned
                };
                Ok(doc.trim().to_string())
            }
            OutputType::YesNo => {
                let lower = cleaned.to_lowercase();
                if lower.starts_with("yes") {
                    Ok("yes".into())
                } else if lower.starts_with("no") {
                    Ok("no".into())
                } else {
                    Err(LLMError::InvalidYesNo(cleaned.to_string()))
                }
            }
        }
    }

    /// Generate with retry logic for malformed outputs
    pub async fn generate_with_retry(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        expected: OutputType,
    ) -> Result<String> {
        for attempt in 0..self.max_retries {
            let response = self.complete(system_prompt, user_prompt).await?;

            match Self::parse_output(&response, expected) {
                Ok(parsed) => return Ok(parsed),
                Err(e) if attempt < self.max_retries - 1 => {
                    warn!("Parse failed (attempt {}), retrying: {}", attempt + 1, e);
                    continue;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to parse LLM output after {} attempts: {}",
                        self.max_retries,
                        e
                    ));
                }
            }
        }

        Err(anyhow::anyhow!(LLMError::MaxRetriesExceeded))
    }

    /// Generate a document
    pub async fn generate_document(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String> {
        self.generate_with_retry(Some(system_prompt), user_prompt, OutputType::Document)
            .await
    }

    /// Generate a query
    pub async fn generate_query(&self, user_prompt: &str) -> Result<String> {
        self.generate_with_retry(None, user_prompt, OutputType::Query)
            .await
    }

    /// Score relevance (returns 0-3)
    pub async fn score_relevance(&self, prompt: &str) -> Result<u8> {
        let score_str = self
            .generate_with_retry(None, prompt, OutputType::Score)
            .await?;
        Ok(score_str.parse()?)
    }

    /// Score relevance on fine-grained scale (returns 0-100)
    pub async fn score_fine_grained(&self, prompt: &str) -> Result<u8> {
        let score_str = self
            .generate_with_retry(None, prompt, OutputType::FineScore)
            .await?;
        Ok(score_str.parse()?)
    }

    /// Parse and validate a range score
    fn parse_range_score(raw: &str, min: u16, max: u16) -> Result<u16, LLMError> {
        let cleaned = raw.trim();
        // Extract all digits
        let num_str: String = cleaned
            .chars()
            .filter(|c| c.is_ascii_digit())
            .take(5) // Max 5 digits for u16
            .collect();
        let score: u16 = num_str
            .parse()
            .map_err(|_| LLMError::InvalidRangeScore(cleaned.to_string(), min, max))?;
        if score < min || score > max {
            return Err(LLMError::InvalidRangeScore(cleaned.to_string(), min, max));
        }
        Ok(score)
    }

    /// Score relevance on custom range scale (returns min-max)
    pub async fn score_range(&self, prompt: &str, min: u16, max: u16) -> Result<u16> {
        for attempt in 0..self.max_retries {
            let response = self.complete(None, prompt).await?;

            match Self::parse_range_score(&response, min, max) {
                Ok(score) => return Ok(score),
                Err(e) if attempt < self.max_retries - 1 => {
                    warn!("Parse failed (attempt {}), retrying: {}", attempt + 1, e);
                    continue;
                }
                Err(e) => {
                    return Err(anyhow::anyhow!(
                        "Failed to parse range score after {} attempts: {}",
                        self.max_retries,
                        e
                    ));
                }
            }
        }

        Err(anyhow::anyhow!(LLMError::MaxRetriesExceeded))
    }

    /// Yes/No validation
    pub async fn validate_yes_no(&self, prompt: &str) -> Result<bool> {
        let response = self
            .generate_with_retry(None, prompt, OutputType::YesNo)
            .await?;
        Ok(response == "yes")
    }

    /// Generate a topic
    pub async fn generate_topic(&self, prompt: &str) -> Result<String> {
        self.generate_with_retry(None, prompt, OutputType::Topic)
            .await
    }

    /// Detect language from text samples
    /// Returns the detected language name (e.g., "French", "German", "Japanese")
    pub async fn detect_language(&self, prompt: &str) -> Result<String> {
        // Use Topic output type since we expect a single word/short phrase
        self.generate_with_retry(None, prompt, OutputType::Topic)
            .await
    }
}


/// Multi-endpoint LLM provider that round-robins requests across multiple base URLs or model instances
/// Useful for running multiple LMStudio servers on different ports
///
/// NOTE: LMStudio processes requests sequentially on a single GPU. For true parallel throughput,
/// you need multiple LMStudio server processes on different ports, or use a backend that supports
/// concurrent batch inference (vLLM, TGI, llama.cpp server with multiple slots).
pub struct MultiEndpointProvider {
    providers: Vec<LLMProvider>,
    counter: AtomicUsize,
}

impl MultiEndpointProvider {
    /// Create a new multi-endpoint provider
    /// base_urls can be comma-separated or a single URL
    pub fn new(base_urls: &str, api_key: &str, model: &str, max_retries: u32) -> Result<Self> {
        let urls: Vec<&str> = base_urls.split(',').map(|s| s.trim()).collect();

        if urls.is_empty() {
            anyhow::bail!("At least one base URL is required");
        }

        let mut providers = Vec::with_capacity(urls.len());
        for url in &urls {
            let config = LLMProviderConfig {
                base_url: url.to_string(),
                api_key: api_key.to_string(),
                model: model.to_string(),
                max_retries,
            };
            providers.push(LLMProvider::new(config)?);
        }

        if providers.len() > 1 {
            info!("Created multi-endpoint provider with {} endpoints", providers.len());
        }

        Ok(Self {
            providers,
            counter: AtomicUsize::new(0),
        })
    }

    /// Create a multi-model provider with different model identifiers
    /// Useful for round-robining across multiple loaded model instances in LMStudio
    /// model_ids should be comma-separated (e.g., "instance-1,instance-2,instance-3")
    pub fn new_multi_model(
        base_url: &str,
        api_key: &str,
        model_ids: &str,
        max_retries: u32,
    ) -> Result<Self> {
        let models: Vec<&str> = model_ids.split(',').map(|s| s.trim()).collect();

        if models.is_empty() {
            anyhow::bail!("At least one model identifier is required");
        }

        let mut providers = Vec::with_capacity(models.len());
        for model in &models {
            let config = LLMProviderConfig {
                base_url: base_url.to_string(),
                api_key: api_key.to_string(),
                model: model.to_string(),
                max_retries,
            };
            providers.push(LLMProvider::new(config)?);
        }

        if providers.len() > 1 {
            info!(
                "Created multi-model provider with {} model instances",
                providers.len()
            );
        }

        Ok(Self {
            providers,
            counter: AtomicUsize::new(0),
        })
    }

    /// Get the next provider in round-robin fashion
    fn next_provider(&self) -> &LLMProvider {
        let idx = self.counter.fetch_add(1, Ordering::Relaxed) % self.providers.len();
        &self.providers[idx]
    }

    /// Get number of endpoints
    pub fn endpoint_count(&self) -> usize {
        self.providers.len()
    }

    /// Send a completion request (round-robin across endpoints)
    pub async fn complete(&self, system_prompt: Option<&str>, user_prompt: &str) -> Result<String> {
        self.next_provider().complete(system_prompt, user_prompt).await
    }

    /// Generate with retry logic
    pub async fn generate_with_retry(
        &self,
        system_prompt: Option<&str>,
        user_prompt: &str,
        expected: OutputType,
    ) -> Result<String> {
        self.next_provider().generate_with_retry(system_prompt, user_prompt, expected).await
    }

    /// Generate a document
    pub async fn generate_document(&self, system_prompt: &str, user_prompt: &str) -> Result<String> {
        self.next_provider().generate_document(system_prompt, user_prompt).await
    }

    /// Generate a query
    pub async fn generate_query(&self, user_prompt: &str) -> Result<String> {
        self.next_provider().generate_query(user_prompt).await
    }

    /// Score relevance (returns 0-3)
    pub async fn score_relevance(&self, prompt: &str) -> Result<u8> {
        self.next_provider().score_relevance(prompt).await
    }

    /// Score relevance on fine-grained scale (returns 0-100)
    pub async fn score_fine_grained(&self, prompt: &str) -> Result<u8> {
        self.next_provider().score_fine_grained(prompt).await
    }

    /// Score relevance on custom range scale (returns min-max)
    pub async fn score_range(&self, prompt: &str, min: u16, max: u16) -> Result<u16> {
        self.next_provider().score_range(prompt, min, max).await
    }

    /// Yes/No validation
    pub async fn validate_yes_no(&self, prompt: &str) -> Result<bool> {
        self.next_provider().validate_yes_no(prompt).await
    }

    /// Generate a topic
    pub async fn generate_topic(&self, prompt: &str) -> Result<String> {
        self.next_provider().generate_topic(prompt).await
    }

    /// Get first provider (for backwards compatibility)
    pub fn first(&self) -> &LLMProvider {
        &self.providers[0]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_score() {
        assert_eq!(LLMProvider::parse_output("3", OutputType::Score).unwrap(), "3");
        assert_eq!(LLMProvider::parse_output("2\n", OutputType::Score).unwrap(), "2");
        assert_eq!(
            LLMProvider::parse_output("Score: 1", OutputType::Score).unwrap(),
            "1"
        );
        assert!(LLMProvider::parse_output("5", OutputType::Score).is_err());
        assert!(LLMProvider::parse_output("abc", OutputType::Score).is_err());
    }

    #[test]
    fn test_parse_query() {
        assert_eq!(
            LLMProvider::parse_output("How do I cook pasta?", OutputType::Query).unwrap(),
            "How do I cook pasta?"
        );
        assert_eq!(
            LLMProvider::parse_output("\"quoted query\"", OutputType::Query).unwrap(),
            "quoted query"
        );
        assert_eq!(
            LLMProvider::parse_output("First line\nSecond line", OutputType::Query).unwrap(),
            "First line"
        );
    }

    #[test]
    fn test_parse_yes_no() {
        assert_eq!(
            LLMProvider::parse_output("yes", OutputType::YesNo).unwrap(),
            "yes"
        );
        assert_eq!(
            LLMProvider::parse_output("Yes, it is relevant.", OutputType::YesNo).unwrap(),
            "yes"
        );
        assert_eq!(
            LLMProvider::parse_output("no", OutputType::YesNo).unwrap(),
            "no"
        );
        assert!(LLMProvider::parse_output("maybe", OutputType::YesNo).is_err());
    }
}
