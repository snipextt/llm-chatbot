mod cli;
mod completion;
mod routes;
mod schemas;
mod splitter;
mod util;

use crate::cli::{parse_args, start_repl, Mode};
use crate::schemas::EncodingRequest;
use crate::util::spawn_embedding_model;
use std::net::SocketAddr;

use axum::routing::get;
use axum::Router;
use splitter::TextSplitter;

use sqlx::postgres::PgPoolOptions;

use util::store_data;

use crate::routes::{answer::answer_handler, ws::ws_handler};
use crate::schemas::AppState;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().unwrap();
    let db_uri = std::env::var("PG_URI").expect("DATABASE_URL is not set");
    let config = parse_args();
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<EncodingRequest>();
    let state = AppState {
        pool: PgPoolOptions::new()
            .max_connections(10)
            .connect(db_uri.as_str())
            .await
            .unwrap(),
        tx: tx.clone(),
        req_client: reqwest::Client::new(),
    };
    spawn_embedding_model(rx);
    store_data(state.pool.clone(), tx, &config).await.unwrap();

    match config.mode {
        Mode::Offline => {
            start_repl(state.tx, state.pool, state.req_client).await;
        }
        Mode::Online => {
            let app = Router::new()
                .route("/answer", get(answer_handler))
                .route("/ws", get(ws_handler))
                .with_state(state);

            let addr = SocketAddr::from(([127, 0, 0, 1], 3000));
            println!("Starting server on {addr:?}");
            axum::Server::bind(&addr)
                .serve(app.into_make_service())
                .await
                .unwrap();
        }
    }
}
