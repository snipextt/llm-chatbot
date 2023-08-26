mod cli;
mod parser;

use cli::parse_args;
use futures_util::future::join_all;
use parser::EmbeddingMessage;
use parser::Parser;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use sqlx::postgres::PgPoolOptions;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::BufReader;
use std::io::Read;
use std::path::PathBuf;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;
use tokio::{spawn, task};

#[derive(Debug, Default, Clone)]
enum MessageType {
    #[default]
    Message,
    Close,
}

#[derive(Debug, Default, Clone)]
pub struct Message<T> {
    message_type: MessageType,
    message: Option<T>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    dotenvy::dotenv()?;
    let db_uri = std::env::var("PG_URI").expect("DATABASE_URL is not set");
    let pool = PgPoolOptions::new()
        .max_connections(10)
        .connect(db_uri.as_str())
        .await?;

    let config = parse_args();
    let dir = match fs::read_dir(&config.path) {
        Ok(dir) => dir,
        Err(e) => return Err(e)?,
    };
    if fs::metadata(&config.index).is_err() {
        fs::File::create(&config.index)?;
    }
    let index: HashMap<String, u64> =
        serde_json::from_reader(BufReader::new(fs::File::open(config.index)?))?;
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Message<EmbeddingMessage>>();
    let mut send_tasks = Vec::new();
    let mut rcv_tasks = Vec::new();
    for file in dir {
        let mut index = index.clone();
        let tx = tx.clone();
        send_tasks.push(task::spawn_blocking(move || {
            let path = file.unwrap().path();
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()
                    .unwrap();
            let last_modified = get_last_modified(&path).unwrap();
            if last_modified > index.get(path.to_str().unwrap()).unwrap_or(&0).to_owned() {
                let mut file = BufReader::new(fs::File::open(path.clone()).unwrap());
                let mut input = String::new();
                file.read_to_string(&mut input).unwrap();
                let input: Vec<char> = input.chars().collect();
                let parser = Parser::new(&model, input.as_slice(), 2048 * 4);
                for embeddings in parser {
                    tx.send(Message {
                        message: Some(embeddings),
                        ..Default::default()
                    })
                    .unwrap();
                }
                index.insert(
                    path.to_str().unwrap().to_string(),
                    SystemTime::now()
                        .duration_since(UNIX_EPOCH)
                        .expect("Is it running back?")
                        .as_secs(),
                );
            }
        }));
    }
    rcv_tasks.push(spawn(async move {
        while let Some(msg) = rx.recv().await {
            match msg.message_type {
                MessageType::Close => rx.close(),
                MessageType::Message => {
                    sqlx::query("INSERT INTO documents (embedding, raw) VALUES ($1, $2)")
                        .bind(msg.clone().message.unwrap().embeddings)
                        .bind(msg.message.unwrap().raw)
                        .execute(&pool)
                        .await
                        .unwrap();
                }
            }
        }
    }));
    rcv_tasks.push(spawn(async move {
        join_all(send_tasks).await;
        tx.send(Message {
            message_type: MessageType::Close,
            ..Default::default()
        })
        .unwrap();
    }));
    join_all(rcv_tasks).await;
    Ok(())
}

fn get_last_modified(path: &PathBuf) -> Result<u64, Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs())
}
