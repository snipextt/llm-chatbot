use sqlx::{Pool, Postgres};
use std::{
    env,
    io::{self, Write},
};

use tokio::sync::mpsc::UnboundedSender;

use crate::{
    completion::Completion,
    schemas::{DocumentRef, EncodingRequest, OpenAiCompletionMessage},
    util::{generate_embedding_for_text, sort_embeddings},
};

#[derive(Debug, Default, PartialEq)]
pub enum Mode {
    #[default]
    Offline,
    Online,
}

#[derive(Default, Debug)]
pub struct Config {
    pub path: String,
    pub mode: Mode,
    pub index: String,
}

pub fn parse_args() -> Config {
    let path_from_env = env::var("DATA_DIR").unwrap_or("".to_string());
    let mut config = Config::default();
    let index_path = env::var("INDEX_PATH").unwrap_or("".to_string());
    let args: Vec<String> = env::args().skip(1).collect();
    args.iter().for_each(|arg| {
        let (key, mut value) = arg.split_at(arg.find("=").unwrap_or(0));
        value = value.trim_start_matches("=");
        if key == "--data-dir" {
            config.path = value.to_string();
        } else if key == "--index-path" {
            config.index = value.to_string();
        } else if key == "--mode" {
            if value == "online" {
                config.mode = Mode::Online;
            }
        }
    });
    if config.mode == Mode::Offline {
        if config.path.is_empty() {
            if !path_from_env.is_empty() {
                config.path = path_from_env;
            } else {
                panic!("--data-dir is required or set PATH in env");
            }
        }

        if config.index.is_empty() {
            if !index_path.is_empty() {
                config.index = index_path;
            } else {
                panic!("--index-path is required or set INDEX_PATH in env");
            }
        }
    }

    config
}

pub async fn start_repl(
    tx: UnboundedSender<EncodingRequest>,
    pool: Pool<Postgres>,
    req_client: reqwest::Client,
) {
    println!("Hi! Do you have any questions?");
    let mut history = Vec::new();
    loop {
        let mut lock = io::stdout().lock();
        lock.write("> ".as_bytes()).unwrap();
        lock.flush().unwrap();
        drop(lock);
        let mut prompt = String::new();
        io::stdin().read_line(&mut prompt).unwrap();
        let embeddings = generate_embedding_for_text(tx.clone(), prompt.clone())
            .await
            .expect("Something went wrong! Unable to generate embeddings.");
        let embeddings = sqlx::query_as::<_, DocumentRef>(
        "SELECT *, embedding <-> $1 as relevence FROM documents ORDER BY embedding <-> $1 LIMIT 6",
    )
    .bind(&embeddings)
    .fetch_all(&pool)
    .await
    .unwrap();
        let embeddings = sort_embeddings(embeddings);
        let completion = Completion::new(embeddings.join("\n"), &req_client, history.clone());
        let answer = completion
            .generate(prompt.clone())
            .await
            .expect("Unable to generate message");
        history.push(OpenAiCompletionMessage {
            role: crate::schemas::OpenAiCompletionRole::User,
            content: prompt,
        });
        history.push(answer.clone());
        println!("{answer}", answer = answer.content.trim_start_matches("\n"));
    }
}
