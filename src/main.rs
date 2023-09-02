mod cli;
mod lexer;
mod schemas;
mod util;

use std::net::SocketAddr;

use axum::extract::Path;
use axum::extract::State;
use axum::routing::get;
use axum::Router;
use lexer::Lexer;

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
use schemas::EncodingRequest;
use sqlx::postgres::PgPoolOptions;
use sqlx::Pool;
use sqlx::Postgres;

use tokio::sync::mpsc::UnboundedReceiver;
use tokio::sync::mpsc::UnboundedSender;
use tokio::task::spawn_blocking;
use util::store_data;

#[derive(Clone)]
struct AppState {
    pool: Pool<Postgres>,
    tx: UnboundedSender<EncodingRequest>,
}

#[tokio::main]
async fn main() {
    dotenvy::dotenv().unwrap();
    let db_uri = std::env::var("PG_URI").expect("DATABASE_URL is not set");
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<EncodingRequest>();
    let state = AppState {
        pool: PgPoolOptions::new()
            .max_connections(10)
            .connect(db_uri.as_str())
            .await
            .unwrap(),
        tx: tx.clone(),
    };
    create_embedding_model(rx);
    store_data(state.pool.clone(), tx).await.unwrap();

    let app = Router::new()
        .route("/answer/:question", get(answer_handler))
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
    axum::Server::bind(&addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

#[derive(sqlx::FromRow, Debug)]
struct Document {
    embedding: Vec<f32>,
    raw: String,
}

async fn answer_handler(
    Path(question): Path<String>,
    State(state): State<AppState>,
) -> &'static str {
    let (tx, rx) = tokio::sync::oneshot::channel();
    let _ = state.tx.send(EncodingRequest {
        raw: vec![question],
        tx,
    });
    let embeddings = &rx.await.unwrap()[0];
    let value =
        sqlx::query_as::<_, Document>("SELECT * FROM documents ORDER BY embedding <-> $1 LIMIT 8")
            .bind(embeddings)
            .fetch_all(&state.pool)
            .await
            .unwrap();

    "Hello, World!"
}

fn create_embedding_model(mut rx: UnboundedReceiver<EncodingRequest>) {
    spawn_blocking(move || {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()
            .unwrap();
        while let Some(msg) = rx.blocking_recv() {
            let embeddings = model.encode(&msg.raw).expect("Failed to encode");
            let _ = msg.tx.send(embeddings);
        }
    });
}
