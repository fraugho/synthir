//! synthir - Synthetic IR Dataset Generator
//!
//! A Rust CLI tool that generates high-quality information retrieval datasets
//! representing real user queries and documents.

pub mod benchmark;
pub mod checkpoint;
pub mod config;
pub mod generation;
pub mod llm;
pub mod meta;
pub mod mining;
pub mod output;
pub mod topics;
pub mod utils;

pub use benchmark::*;
pub use checkpoint::*;
pub use config::*;
pub use generation::*;
pub use llm::*;
pub use meta::*;
pub use mining::*;
pub use output::*;
pub use topics::*;
pub use utils::*;
