use crate::parser::EmbeddingMessage;

pub struct EncodingRequest {
    pub raw: String,
    pub tx: UnboundedSender<EmbeddingMessage>,
}
