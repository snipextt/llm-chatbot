use std::error::Error;

use serde_json::{from_value, Value};

use crate::schemas::{
    OpenAiCompletionMessage, OpenAiCompletionModel, OpenAiCompletionRequest, OpenAiCompletionRole,
};

pub struct Completion<'a> {
    context: String,
    client: &'a reqwest::Client,
    history: Vec<OpenAiCompletionMessage>,
}

impl<'a> Completion<'a> {
    pub fn new(
        context: String,
        client: &'a reqwest::Client,
        history: Vec<OpenAiCompletionMessage>,
    ) -> Self {
        Completion {
            context,
            client,
            history,
        }
    }

    pub async fn generate(
        &self,
        prompt: String,
    ) -> Result<OpenAiCompletionMessage, Box<dyn Error>> {
        const OPEN_AI_URI: &str = "https://api.openai.com/v1/chat/completions";
        let token = std::env::var("OPEN_AI_TOKEN").unwrap();
        let mut messages = vec![
            OpenAiCompletionMessage {
                role: OpenAiCompletionRole::System,
                content: format!("You are a chatbot assistant"),
            },
            OpenAiCompletionMessage {
                role: OpenAiCompletionRole::System,
                content: self.context.clone(),
            },
        ];
        for message in self.history.clone() {
            messages.push(message.to_owned());
        }
        if self.history.len() == 0 {
            messages.push(OpenAiCompletionMessage {
                role: OpenAiCompletionRole::User,
                content: format!("Hello"),
            });
            messages.push(OpenAiCompletionMessage {
                role: OpenAiCompletionRole::User,
                content: format!("Hi, I am a chat assistant. How many i help you today?"),
            });
        }
        messages.push(OpenAiCompletionMessage {
            role: OpenAiCompletionRole::User,
            content: prompt,
        });
        let response: Value = self
            .client
            .post(OPEN_AI_URI)
            .header("Authorization", format!("Bearer {token}"))
            .json(&OpenAiCompletionRequest {
                model: OpenAiCompletionModel::GPT35TURBO,
                messages,
            })
            .send()
            .await?
            .json()
            .await?;

        // TODO: Error handling
        let completion: OpenAiCompletionMessage = from_value(
            response
                .get("choices")
                .unwrap()
                .to_owned()
                .as_array()
                .unwrap()
                .get(0)
                .unwrap()
                .get("message")
                .unwrap()
                .clone(),
        )?;

        Ok(completion)
    }
}
