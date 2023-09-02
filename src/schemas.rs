use tokio::sync::oneshot::Sender;

use crate::lexer::EmbeddingMessage;

pub struct EncodingRequest {
    pub raw: Vec<String>,
    pub tx: Sender<Vec<Vec<f32>>>,
}
