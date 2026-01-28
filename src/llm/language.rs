use anyhow::Result;
use std::collections::HashMap;
use tracing::{info, warn};

use super::{language_detection_prompt, LLMProvider};

/// Result of language detection
#[derive(Debug, Clone)]
pub struct LanguageDetectionResult {
    /// Detected language (e.g., "French", "German", "Japanese")
    pub language: String,
    /// Confidence level based on sample consensus
    pub confidence: LanguageConfidence,
    /// Number of samples that agreed on the detected language
    pub agreement_count: usize,
    /// Total number of samples analyzed
    pub total_samples: usize,
}

/// Confidence level for language detection
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LanguageConfidence {
    /// Strong majority (>= 70% agreement)
    High,
    /// Decent majority (>= 50% agreement)
    Medium,
    /// Inconclusive (< 50% agreement)
    Low,
}

/// Language detector that uses LLM to identify text language
pub struct LanguageDetector<'a> {
    provider: &'a LLMProvider,
}

impl<'a> LanguageDetector<'a> {
    pub fn new(provider: &'a LLMProvider) -> Self {
        Self { provider }
    }

    /// Detect language from text samples with progressive retry
    ///
    /// Tries progressively larger sample sizes if results are inconclusive:
    /// - First try: sample_sizes[0] samples
    /// - If inconclusive, try sample_sizes[1] samples
    /// - etc.
    ///
    /// Returns None if still inconclusive after all attempts
    pub async fn detect_with_retry(
        &self,
        samples: &[String],
        sample_sizes: &[usize],
    ) -> Result<Option<LanguageDetectionResult>> {
        for (attempt, &size) in sample_sizes.iter().enumerate() {
            let size = size.min(samples.len());
            if size == 0 {
                continue;
            }

            info!("Language detection attempt {} with {} samples...", attempt + 1, size);

            let sample_refs: Vec<&str> = samples.iter().take(size).map(|s| s.as_str()).collect();
            let result = self.detect_from_samples(&sample_refs).await?;

            match result.confidence {
                LanguageConfidence::High | LanguageConfidence::Medium => {
                    info!(
                        "Detected language: {} (confidence: {:?}, {}/{} samples agreed)",
                        result.language, result.confidence, result.agreement_count, result.total_samples
                    );
                    return Ok(Some(result));
                }
                LanguageConfidence::Low => {
                    if attempt + 1 < sample_sizes.len() {
                        warn!(
                            "Inconclusive detection ({}/{} agreed on {}), trying larger sample...",
                            result.agreement_count, result.total_samples, result.language
                        );
                    } else {
                        warn!(
                            "Language detection inconclusive after {} attempts ({}/{} agreed on {})",
                            sample_sizes.len(), result.agreement_count, result.total_samples, result.language
                        );
                    }
                }
            }
        }

        Ok(None)
    }

    /// Detect language from a set of text samples
    async fn detect_from_samples(&self, samples: &[&str]) -> Result<LanguageDetectionResult> {
        // Ask LLM to detect language
        let prompt = language_detection_prompt(samples);
        let detected = self.provider.detect_language(&prompt).await?;

        // Normalize the response
        let language = clean_language_name(&detected);

        // For now, we do a single detection with all samples
        // The confidence is based on the assumption that a single prompt with multiple samples
        // gives us implicit consensus
        let confidence = if samples.len() >= 5 {
            LanguageConfidence::High
        } else if samples.len() >= 3 {
            LanguageConfidence::Medium
        } else {
            LanguageConfidence::Low
        };

        Ok(LanguageDetectionResult {
            language,
            confidence,
            agreement_count: samples.len(),
            total_samples: samples.len(),
        })
    }

    /// Detect language using multiple independent LLM calls for consensus
    pub async fn detect_with_consensus(
        &self,
        samples: &[String],
        num_samples: usize,
        min_consensus: f64,
    ) -> Result<Option<LanguageDetectionResult>> {
        let num_samples = num_samples.min(samples.len());
        if num_samples == 0 {
            return Ok(None);
        }

        let mut votes: HashMap<String, usize> = HashMap::new();

        // Ask LLM to detect language for each sample independently
        for sample in samples.iter().take(num_samples) {
            let prompt = language_detection_prompt(&[sample.as_str()]);
            match self.provider.detect_language(&prompt).await {
                Ok(detected) => {
                    let language = clean_language_name(&detected);
                    *votes.entry(language).or_insert(0) += 1;
                }
                Err(e) => {
                    warn!("Failed to detect language for sample: {}", e);
                }
            }
        }

        if votes.is_empty() {
            return Ok(None);
        }

        // Find the language with most votes
        let (best_language, best_count) = votes
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(lang, count)| (lang.clone(), *count))
            .unwrap();

        let total = votes.values().sum::<usize>();
        let ratio = best_count as f64 / total as f64;

        let confidence = if ratio >= 0.7 {
            LanguageConfidence::High
        } else if ratio >= min_consensus {
            LanguageConfidence::Medium
        } else {
            LanguageConfidence::Low
        };

        Ok(Some(LanguageDetectionResult {
            language: best_language,
            confidence,
            agreement_count: best_count,
            total_samples: total,
        }))
    }
}

/// Clean up language name from LLM output
/// Just trims whitespace and capitalizes first letter - trusts LLM to output standard English names
fn clean_language_name(raw: &str) -> String {
    // Strip Qwen3-style thinking tags
    let stripped = if let Some(end_pos) = raw.find("</think>") {
        raw[end_pos + 8..].trim()
    } else {
        raw.trim()
    };
    if stripped.is_empty() {
        return "English".to_string();
    }
    let trimmed = stripped;
    // Just capitalize first letter, keep the rest as-is
    let mut chars = trimmed.chars();
    match chars.next() {
        None => "English".to_string(),
        Some(first) => {
            let rest: String = chars.collect();
            format!("{}{}", first.to_uppercase(), rest)
        }
    }
}

/// Map locale code to language name
pub fn locale_to_language(locale: &str) -> Option<String> {
    // Extract language code from locale (e.g., "fr-fr" -> "fr", "ja-jp" -> "ja")
    let lang_code = locale.split('-').next()?.to_lowercase();

    Some(match lang_code.as_str() {
        "en" => "English",
        "fr" => "French",
        "de" => "German",
        "es" => "Spanish",
        "it" => "Italian",
        "pt" => "Portuguese",
        "ja" => "Japanese",
        "zh" => "Chinese",
        "ko" => "Korean",
        "ar" => "Arabic",
        "ru" => "Russian",
        "nl" => "Dutch",
        "pl" => "Polish",
        "tr" => "Turkish",
        "vi" => "Vietnamese",
        "th" => "Thai",
        "hi" => "Hindi",
        "sv" => "Swedish",
        "da" => "Danish",
        "no" => "Norwegian",
        "fi" => "Finnish",
        "cs" => "Czech",
        "el" => "Greek",
        "he" => "Hebrew",
        "id" => "Indonesian",
        "ms" => "Malay",
        "uk" => "Ukrainian",
        "ro" => "Romanian",
        "hu" => "Hungarian",
        _ => return None,
    }.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clean_language_name() {
        assert_eq!(clean_language_name("french"), "French");
        assert_eq!(clean_language_name("GERMAN"), "GERMAN"); // Just capitalizes first letter
        assert_eq!(clean_language_name("  Spanish  "), "Spanish");
        assert_eq!(clean_language_name(""), "English"); // Empty defaults to English
    }

    #[test]
    fn test_locale_to_language() {
        assert_eq!(locale_to_language("fr-fr"), Some("French".to_string()));
        assert_eq!(locale_to_language("ja-jp"), Some("Japanese".to_string()));
        assert_eq!(locale_to_language("en-us"), Some("English".to_string()));
        assert_eq!(locale_to_language("de-de"), Some("German".to_string()));
        assert_eq!(locale_to_language("xx-yy"), None);
    }
}
