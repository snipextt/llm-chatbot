mod cli;
mod parser;
mod schemas;
mod util;

use parser::Lexer;

use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsBuilder;
use rust_bert::pipelines::sentence_embeddings::SentenceEmbeddingsModelType;
use schemas::EncodingRequest;
use sqlx::postgres::PgPoolOptions;
use sqlx::Pool;
use sqlx::Postgres;

use tokio::task::spawn_blocking;
use util::store_data;

struct AppState {
    pool: Pool<Postgres>,
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
    };
    store_data(state.pool.clone(), tx).await.unwrap();
    let embedding_thread = spawn_blocking(move || {
        let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
            .create_model()
            .unwrap();
    });
}
