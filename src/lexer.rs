use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModel;
use std::cmp::min;

#[derive(Debug, Clone, Default)]
pub struct EmbeddingMessage {
    pub embeddings: Vec<f32>,
    pub raw: String,
}

pub struct Lexer<'a> {
    source: &'a [char],
    context_length: usize, // 128 * 4 -> where 4 is the number of chars per token
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a [char], context_length: usize) -> Self {
        Lexer {
            source,
            context_length,
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = String;

    fn next(&mut self) -> Option<Self::Item> {
        let mut sl = String::new();
        loop {
            if self.source.len() == 0 {
                return None;
            }
            sl.push(self.source[0]);
            self.source = &self.source[1..];
            if sl.ends_with("\n\n") {
                sl = sl.trim_end_matches("\n").to_string();
                break;
            }
            if sl.ends_with(".") {
                let av_len = self.context_length - sl.len();
                let remaining_slice = &self.source[0..min(av_len, self.source.len())];
                if remaining_slice
                    .iter()
                    .find(|v| v.to_string() == ".")
                    .is_none()
                    && av_len > self.source.len()
                {
                    break;
                }
            }
            if sl.ends_with("\n") {
                sl.pop();
                sl.push_str(" ");
            }
            if sl.len() > self.context_length {
                break;
            }
        }
        Some(String::from(sl))
    }
}
