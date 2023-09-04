use serde::{Deserialize, Serialize};
use sqlx::{Pool, Postgres};
use tokio::sync::{mpsc::UnboundedSender, oneshot::Sender};

pub struct EncodingRequest {
    pub raw: Vec<String>,
    pub tx: Sender<Vec<Vec<f32>>>,
}

#[derive(Debug, sqlx::FromRow, Clone, Default)]
pub struct DocumentRef {
    pub embedding: Vec<f32>,
    pub raw: String,
    pub relevence: Option<f32>,
    pub doc_ref: String,
    pub segment: i64,
}

#[derive(Clone)]
pub struct AppState {
    pub pool: Pool<Postgres>,
    pub tx: UnboundedSender<EncodingRequest>,
    pub req_client: reqwest::Client,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum OpenAiCompletionRole {
    #[serde(rename = "system")]
    System,
    #[serde(rename = "user")]
    User,
    #[serde(rename = "assistant")]
    Assistant,
}

#[derive(Serialize, Deserialize)]
pub enum OpenAiCompletionModel {
    #[serde(rename = "gpt-3.5-turbo")]
    GPT35TURBO,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct OpenAiCompletionMessage {
    pub role: OpenAiCompletionRole,
    pub content: String,
}

#[derive(Serialize)]
pub struct OpenAiCompletionRequest {
    pub model: OpenAiCompletionModel,
    pub messages: Vec<OpenAiCompletionMessage>,
}
