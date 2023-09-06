use axum::{
    extract::{Query, State},
    http::StatusCode,
};
use serde::{Deserialize, Serialize};

use crate::{
    completion::Completion,
    schemas::{AppState, DocumentRef},
    util::{generate_embedding_for_text, sort_embeddings},
};

#[derive(Serialize, Deserialize)]
pub struct Question {
    question: String,
}

pub async fn answer_handler(
    query: Query<Question>,
    State(state): State<AppState>,
) -> Result<String, StatusCode> {
    let embeddings = generate_embedding_for_text(state.tx, query.question.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    let segments = sqlx::query_as::<_, DocumentRef>(
        "SELECT *, embedding <-> $1 as relevence FROM documents ORDER BY relevence LIMIT 4",
    )
    .bind(&embeddings)
    .fetch_all(&state.pool)
    .await
    .unwrap();
    let segments = sort_embeddings(segments);
    let completion = Completion::new(segments.join("\n"), &state.req_client, vec![]);
    let answer = completion
        .generate(query.question.clone())
        .await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?;
    Ok(answer.content)
}
