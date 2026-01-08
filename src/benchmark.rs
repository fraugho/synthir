//! Benchmark mode for finding optimal concurrency settings

use crate::llm::{LLMProvider, LLMProviderConfig};
use anyhow::Result;
use futures::stream::{self, StreamExt};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Semaphore;

/// Result of a single benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub concurrency: usize,
    pub total_requests: usize,
    pub total_duration: Duration,
    pub requests_per_second: f64,
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
    pub total_tokens: usize,
    pub tokens_per_second: f64,
}

/// Configuration for benchmark
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub base_url: String,
    pub api_key: String,
    pub model: String,
    pub concurrency_levels: Vec<usize>,
    pub samples_per_level: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            base_url: "https://api.openai.com/v1".to_string(),
            api_key: String::new(),
            model: "gpt-4".to_string(),
            concurrency_levels: vec![1, 2, 4, 8, 16, 32],
            samples_per_level: 10,
        }
    }
}

/// Simple test prompt for benchmarking
const BENCHMARK_PROMPT: &str = "Generate a single sentence about a random topic. Be creative.";

/// Run latency test (single request, multiple samples)
async fn run_latency_test(
    provider: &LLMProvider,
    samples: usize,
) -> Result<(Vec<Duration>, Vec<usize>)> {
    let mut latencies = Vec::with_capacity(samples);
    let mut token_counts = Vec::with_capacity(samples);

    for _ in 0..samples {
        let start = Instant::now();
        let response = provider.complete(None, BENCHMARK_PROMPT).await?;
        let elapsed = start.elapsed();

        latencies.push(elapsed);
        // Rough token estimate: ~4 chars per token
        token_counts.push(response.len() / 4);
    }

    Ok((latencies, token_counts))
}

/// Run throughput test at a specific concurrency level
async fn run_throughput_test(
    provider: &LLMProvider,
    concurrency: usize,
    total_requests: usize,
) -> Result<BenchmarkResult> {
    let semaphore = Arc::new(Semaphore::new(concurrency));
    let start = Instant::now();

    let futures: Vec<_> = (0..total_requests)
        .map(|_| {
            let sem = semaphore.clone();
            async move {
                let _permit = sem.acquire().await.unwrap();
                let req_start = Instant::now();
                let response = provider.complete(None, BENCHMARK_PROMPT).await;
                let elapsed = req_start.elapsed();
                (elapsed, response)
            }
        })
        .collect();

    let results: Vec<_> = stream::iter(futures)
        .buffer_unordered(concurrency)
        .collect()
        .await;

    let total_duration = start.elapsed();

    // Collect latencies and token counts
    let mut latencies: Vec<Duration> = Vec::new();
    let mut total_tokens = 0usize;

    for (latency, result) in results {
        if let Ok(response) = result {
            latencies.push(latency);
            total_tokens += response.len() / 4; // Rough estimate
        }
    }

    // Sort latencies for percentile calculations
    latencies.sort();

    let total_requests = latencies.len();
    if total_requests == 0 {
        return Ok(BenchmarkResult {
            concurrency,
            total_requests: 0,
            total_duration,
            requests_per_second: 0.0,
            avg_latency_ms: 0.0,
            p50_latency_ms: 0.0,
            p95_latency_ms: 0.0,
            p99_latency_ms: 0.0,
            total_tokens: 0,
            tokens_per_second: 0.0,
        });
    }

    let avg_latency = latencies.iter().map(|d| d.as_millis()).sum::<u128>() / total_requests as u128;
    let p50_idx = total_requests / 2;
    let p95_idx = (total_requests as f64 * 0.95) as usize;
    let p99_idx = (total_requests as f64 * 0.99) as usize;

    let requests_per_second = total_requests as f64 / total_duration.as_secs_f64();
    let tokens_per_second = total_tokens as f64 / total_duration.as_secs_f64();

    Ok(BenchmarkResult {
        concurrency,
        total_requests,
        total_duration,
        requests_per_second,
        avg_latency_ms: avg_latency as f64,
        p50_latency_ms: latencies[p50_idx].as_millis() as f64,
        p95_latency_ms: latencies[p95_idx.min(total_requests - 1)].as_millis() as f64,
        p99_latency_ms: latencies[p99_idx.min(total_requests - 1)].as_millis() as f64,
        total_tokens,
        tokens_per_second,
    })
}

/// Run full benchmark suite
pub async fn run_benchmark(config: BenchmarkConfig) -> Result<Vec<BenchmarkResult>> {
    let provider = LLMProvider::new(LLMProviderConfig {
        base_url: config.base_url.clone(),
        api_key: config.api_key.clone(),
        model: config.model.clone(),
        max_retries: 1, // Fewer retries for benchmarking
    })?;

    println!("\n=== synthir Benchmark ===\n");
    println!("Model: {}", config.model);
    println!("Endpoint: {}", config.base_url);
    println!("Samples per level: {}", config.samples_per_level);
    println!();

    // Run latency test first
    println!("Running latency test (single request)...");
    let (latencies, tokens) = run_latency_test(&provider, config.samples_per_level).await?;

    let mut sorted_latencies: Vec<_> = latencies.iter().map(|d| d.as_millis()).collect();
    sorted_latencies.sort();

    let avg_latency = sorted_latencies.iter().sum::<u128>() / sorted_latencies.len() as u128;
    let p50_idx = sorted_latencies.len() / 2;
    let p95_idx = (sorted_latencies.len() as f64 * 0.95) as usize;
    let p99_idx = (sorted_latencies.len() as f64 * 0.99) as usize;

    println!("\nLatency Test Results (single request):");
    println!("  Mean:  {}ms", avg_latency);
    println!("  P50:   {}ms", sorted_latencies[p50_idx]);
    println!(
        "  P95:   {}ms",
        sorted_latencies[p95_idx.min(sorted_latencies.len() - 1)]
    );
    println!(
        "  P99:   {}ms",
        sorted_latencies[p99_idx.min(sorted_latencies.len() - 1)]
    );

    let avg_tokens: usize = tokens.iter().sum::<usize>() / tokens.len().max(1);
    println!("  Avg tokens/response: ~{}", avg_tokens);
    println!();

    // Run throughput tests
    println!("Running throughput tests...\n");
    println!(
        "{:>11} | {:>8} | {:>10} | {:>12}",
        "Concurrency", "Req/s", "Tokens/s", "Avg Latency"
    );
    println!("{}", "-".repeat(50));

    let mut results = Vec::new();

    for &concurrency in &config.concurrency_levels {
        let result = run_throughput_test(&provider, concurrency, config.samples_per_level).await?;

        let marker = if results
            .last()
            .map(|r: &BenchmarkResult| result.tokens_per_second > r.tokens_per_second * 0.95)
            .unwrap_or(true)
        {
            ""
        } else {
            " <-- diminishing returns"
        };

        println!(
            "{:>11} | {:>8.1} | {:>10.0} | {:>9.0}ms{}",
            concurrency,
            result.requests_per_second,
            result.tokens_per_second,
            result.avg_latency_ms,
            marker
        );

        results.push(result);
    }

    // Find optimal concurrency (best tokens/s with reasonable latency)
    let optimal = results
        .iter()
        .max_by(|a, b| {
            a.tokens_per_second
                .partial_cmp(&b.tokens_per_second)
                .unwrap()
        })
        .unwrap();

    println!();
    println!(
        "Recommendation: Use --concurrency {} for optimal throughput",
        optimal.concurrency
    );
    println!(
        "  ({:.0} tokens/s, {:.1} req/s, {:.0}ms avg latency)",
        optimal.tokens_per_second, optimal.requests_per_second, optimal.avg_latency_ms
    );

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.concurrency_levels, vec![1, 2, 4, 8, 16, 32]);
        assert_eq!(config.samples_per_level, 10);
    }
}
