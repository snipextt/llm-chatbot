mod cli;
use futures_util::future::join_all;

use cli::parse_args;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use sqlx::postgres::PgPoolOptions;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;
use std::time::UNIX_EPOCH;
use tokio::task;

pub struct Message {
    embeddings: Vec<f32>,
    raw: String,
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
    let (tx, rx) = tokio::sync::mpsc::channel::<Message>(100);
    let mut tasks = Vec::new();
    for file in dir {
        let index = index.clone();
        let tx = tx.clone();
        tasks.push(task::spawn_blocking(move || {
            let path = file.unwrap().path();
            let model =
                SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL6V2)
                    .create_model()
                    .unwrap();
            let last_modified = get_last_modified(&path).unwrap();
            if last_modified > index.get(path.to_str().unwrap()).unwrap_or(&0).to_owned() {
                let file = parse_file(path.to_str().unwrap()).unwrap();
                let embeddings = model.encode(&file).unwrap();
                for i in embeddings {
                    tx.blocking_send(Message {
                        embeddings: i.to_vec(),
                        raw: "".to_string(),
                    })
                    .unwrap();
                }
            }
        }));
    }
    join_all(tasks).await;
    Ok(())
}

fn parse_file(path: &str) -> Result<Vec<String>, Box<dyn Error>> {
    let content = fs::read_to_string(path)?;
    Ok(content
        .split("\n")
        .filter_map(|v| {
            let line = v
                .trim_start_matches("\n")
                .trim_end_matches("\n")
                .to_string();
            if line.is_empty() {
                None
            } else {
                Some(line)
            }
        })
        .collect())
}

fn get_last_modified(path: &PathBuf) -> Result<u64, Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs())
}
