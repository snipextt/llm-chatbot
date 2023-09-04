use axum::{
    extract::{Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use crate::{
    completion::Completion,
    schemas::{AppState, DocumentRef, EncodingRequest},
    util::sort_embeddings,
};

#[derive(Serialize, Deserialize)]
pub struct Question {
    question: String,
}

pub async fn answer_handler(
    query: Query<Question>,
    State(state): State<AppState>,
) -> Result<String, StatusCode> {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state.tx.send(EncodingRequest {
        raw: vec![query.question.clone()],
        tx,
    });
    let embeddings = rx.await.map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?[0].clone();
    let embeddings = sqlx::query_as::<_, DocumentRef>(
        "SELECT *, embedding <-> $1 as relevence FROM documents ORDER BY embedding <-> $1 LIMIT 6",
    )
    .bind(&embeddings)
    .fetch_all(&state.pool)
    .await
    .unwrap();
    let embeddings = sort_embeddings(embeddings);
    let completion = Completion::new(embeddings.join("\n"), &state.req_client, vec![]);
    let answer = completion
        .generate(query.question.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;

    Ok(answer.content)
}
