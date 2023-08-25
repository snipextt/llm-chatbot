use std::env;

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
