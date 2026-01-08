use anyhow::{Context, Result};
use async_openai::{
    config::OpenAIConfig,
    types::{
        ChatCompletionRequestMessage, ChatCompletionRequestSystemMessageArgs,
        ChatCompletionRequestUserMessageArgs, CreateChatCompletionRequestArgs,
    },
    Client,
};
use thiserror::Error;
use tracing::{debug, warn};

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
            messages.push(
                ChatCompletionRequestSystemMessageArgs::default()
                    .content(sys)
                    .build()?
                    .into(),
            );
        }

        messages.push(
            ChatCompletionRequestUserMessageArgs::default()
                .content(user_prompt)
                .build()?
                .into(),
        );

        let request = CreateChatCompletionRequestArgs::default()
            .model(&self.model)
            .messages(messages)
            .temperature(0.7)
            .build()?;

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
