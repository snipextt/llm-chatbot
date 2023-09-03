use crate::schemas::Document;
use std::{
    collections::HashMap,
    error::Error,
    fs::{self, DirEntry},
    io::{BufReader, BufWriter, Read},
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{cli::parse_args, schemas::EncodingRequest, TextSplitter};
use futures_util::future::join_all;
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use sqlx::{Pool, Postgres};
use tokio::{
    spawn,
    sync::{
        mpsc::{UnboundedReceiver, UnboundedSender},
        oneshot,
    },
    task::{spawn_blocking, JoinHandle},
};

pub fn parse_entry(
    file: DirEntry,
    tx: UnboundedSender<Document>,
    index: Arc<Mutex<HashMap<String, u64>>>,
    task_list: &mut Vec<JoinHandle<()>>,
    tx_m: UnboundedSender<EncodingRequest>,
) {
    task_list.push(spawn(async move {
        let path = file.path();
        let last_modified = get_last_modified(&path).unwrap();
        let to_read_file = {
            let guard = index.lock().unwrap();
            last_modified > guard.get(path.to_str().unwrap()).unwrap_or(&0).to_owned()
        };
        if to_read_file {
            create_embeddings_from_file(&path, tx_m)
                .await
                .iter()
                .for_each(|embeddings| tx.send(embeddings.clone()).unwrap());
            {
                let mut guard = index.lock().unwrap();
                update_index(&path, &mut guard);
            };
        }
    }));
}

async fn create_embeddings_from_file(
    path: &PathBuf,
    tx_m: UnboundedSender<EncodingRequest>,
) -> Vec<Document> {
    let input = read_chars_form_file(path);
    let parsed_input: Vec<String> =
        TextSplitter::new(input.as_slice(), 512 * 4, Some("\n\n")).collect();
    let (tx, rx) = oneshot::channel();
    let _ = tx_m.send(EncodingRequest {
        raw: parsed_input.clone(),
        tx,
    });
    let embeddings = rx.await.unwrap();
    embeddings
        .iter()
        .enumerate()
        .map(|(i, embedding)| Document {
            embedding: embedding.clone(),
            raw: parsed_input[i].clone(),
            doc_ref: path.to_str().unwrap().to_string(),
            segment: i as i64,
            relevence: None,
        })
        .collect()
}

fn update_index(path: &PathBuf, index: &mut HashMap<String, u64>) {
    index.insert(
        path.to_str().unwrap().to_string(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("Is it running back?")
            .as_secs(),
    );
}

fn read_chars_form_file(path: &PathBuf) -> Vec<char> {
    let mut file = BufReader::new(fs::File::open(path.clone()).unwrap());
    let mut input = String::new();
    file.read_to_string(&mut input).unwrap();
    input.chars().collect()
}

fn get_last_modified(path: &PathBuf) -> Result<u64, Box<dyn Error>> {
    let metadata = fs::metadata(path)?;
    Ok(metadata.modified()?.duration_since(UNIX_EPOCH)?.as_secs())
}

pub async fn store_entries(mut rx: UnboundedReceiver<Document>, pool: Pool<Postgres>) {
    while let Some(msg) = rx.recv().await {
        sqlx::query(
            "INSERT INTO documents (embedding, raw, doc_ref, segment) VALUES ($1, $2, $3, $4)",
        )
        .bind(msg.embedding.clone())
        .bind(msg.raw.clone())
        .bind(msg.doc_ref.clone())
        .bind(msg.segment)
        .execute(&pool)
        .await
        .unwrap();
    }
}

pub async fn store_data(
    pool: Pool<Postgres>,
    tx_m: UnboundedSender<EncodingRequest>,
) -> Result<(), Box<dyn Error>> {
    let config = parse_args();
    if fs::metadata(&config.index).is_err() {
        fs::File::create(&config.index)?;
    }
    let index: Arc<Mutex<HashMap<String, u64>>> = Arc::new(Mutex::new(serde_json::from_reader(
        BufReader::new(fs::File::open(config.index.clone())?),
    )?));
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel::<Document>();
    let mut tasks = Vec::new();
    fs::read_dir(&config.path)?.for_each(|entry| {
        parse_entry(
            entry.unwrap(),
            tx.clone(),
            index.clone(),
            &mut tasks,
            tx_m.clone(),
        );
    });
    drop(tx);
    drop(tx_m);
    tasks.push(spawn(async move {
        store_entries(rx, pool).await;
    }));
    join_all(tasks).await;
    serde_json::to_writer(BufWriter::new(fs::File::create(config.index)?), &index)?;

    Ok(())
}

pub fn euclidean_distance(point1: &Vec<f32>, point2: &Vec<f32>) -> f32 {
    assert_eq!(point1.len(), point2.len());
    point1
        .iter()
        .enumerate()
        .fold(0f32, |acc, (i, v)| acc + (v - point2[i]).powi(2))
        .powf(0.5)
}

pub fn sort_embeddings(embeddings: Vec<Document>) -> Vec<String> {
    let mut documents = Vec::new();
    for doc in embeddings {
        documents.push(doc.clone());
    }
    // TODO: If two documents have similar relevency and have close by segments, further sort them
    // in increasing order of segment
    documents.sort_by(|a, b| {
        b.relevence
            .unwrap()
            .partial_cmp(&a.relevence.unwrap())
            .unwrap()
    });
    documents.iter().map(|v| v.raw.clone()).collect()
}

pub fn spawn_embedding_model(mut rx: UnboundedReceiver<EncodingRequest>) {
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
