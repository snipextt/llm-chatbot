use axum::{
    extract::{
        ws::{Message, WebSocketUpgrade},
        State,
    },
    response::IntoResponse,
};
use futures_util::{SinkExt, StreamExt};

use crate::{
    completion::Completion,
    schemas::{AppState, DocumentRef, OpenAiCompletionMessage},
    util::{generate_embedding_for_text, sort_embeddings},
};

pub async fn ws_handler(ws: WebSocketUpgrade, State(state): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(|mut socket| async move {
        if socket.send(Message::Ping(vec![])).await.is_err() {
            return;
        }
        let (mut tx, mut rcv) = socket.split();
        let mut history: Vec<OpenAiCompletionMessage> = Vec::new();
        while let Some(Ok(msg)) = rcv.next().await {
            match msg {
                Message::Text(msg) => {
                    let embeddings =
                        match generate_embedding_for_text(state.tx.clone(), msg.clone()).await {
                            Ok(embedding) => embedding,
                            Err(_) => return,
                        };
                    let segments = sqlx::query_as::<_, DocumentRef>(
                        "SELECT *, embedding <-> $1 as relevence FROM documents ORDER BY relevence LIMIT 4",
                    )
                    .bind(&embeddings)
                    .fetch_all(&state.pool)
                    .await
                    .unwrap();
                    let segments = sort_embeddings(segments);
                    let completion =
                        Completion::new(segments.join("\n"), &state.req_client, history.clone());
                    let answer = match completion.generate(msg.clone()).await {
                        Ok(completion) => completion,
                        Err(_) => return,
                    };
                    history.push(OpenAiCompletionMessage {
                        role: crate::schemas::OpenAiCompletionRole::User,
                        content: msg,
                    });
                    history.push(answer.clone());
                    tx.send(Message::Text(answer.content)).await.unwrap();
                }
                Message::Ping(_) => continue,
                Message::Pong(_) => continue,
                _ => {
                    return;
                }
            }
        }
    })
}
