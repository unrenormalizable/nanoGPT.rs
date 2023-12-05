use crate::{
    logger::*,
    data::{ NanoGptBatch, NanoGptBatcher, NanoGptDataset},
};
use burn::{
    self,
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    nn::loss::CrossEntropyLoss,
    optim::AdamConfig,
    record::CompactRecorder,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
};
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub struct ArgsInfo<'a> {
    pub input_file: &'a str,
    pub output_folder: &'a str,
}

#[derive(Config)]
pub struct TrainingConfig {
    // TODO: Clean this up.
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 64)]
    pub batch_size: usize,
    #[config(default = 4)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

pub fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: B::Device, logger: &dyn NanoGptLogger) {
    std::fs::create_dir_all(artifact_dir).ok();
    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Save without error");

    B::seed(config.seed);

    let batcher_train = NanoGptBatcher::<B>::new(device.clone());
    //let batcher_valid = NanoGptBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(NanoGptDataset::train());

    let dataloader = dataloader_train;
    let mut iterator = dataloader.iter();
    while let Some(item) = iterator.next() {
        logger.info(&format!("??? {item:?}"));
    }

    //let dataloader_test = DataLoaderBuilder::new(batcher_valid)
    //    .batch_size(config.batch_size)
    //    .shuffle(config.seed)
    //    .num_workers(config.num_workers)
    //    .build(NanoGptDataset::test());

    //let learner = LearnerBuilder::new(artifact_dir)
    //    .metric_train_numeric(AccuracyMetric::new())
    //    .metric_valid_numeric(AccuracyMetric::new())
    //    .metric_train_numeric(LossMetric::new())
    //    .metric_valid_numeric(LossMetric::new())
    //    .with_file_checkpointer(CompactRecorder::new())
    //    .devices(vec![device])
    //    .num_epochs(config.num_epochs)
    //    .build(
    //        config.model.init::<B>(),
    //        config.optimizer.init(),
    //        config.learning_rate,
    //    );

    //let model_trained = learner.fit(dataloader_train, dataloader_test);

    //model_trained
    //    .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
    //    .expect("Failed to save trained model");
}

pub fn run(args_info: &ArgsInfo, logger: &dyn NanoGptLogger) -> Result<(), String> {
    logger.info(&format!(
        "Starting with {:?},  {:?}.",
        args_info.input_file, args_info.output_folder
    ));

    let mut file = File::open(args_info.input_file).map_err(|e| e.to_string())?;
    let mut contents = String::new();

    file.read_to_string(&mut contents)
        .map_err(|e| e.to_string())?;

    logger.info(&format!("# of characters {:?}", contents.len()));
    let s: std::collections::HashSet<char> = contents.chars().collect();
    let mut v = Vec::from_iter(s);
    v.sort();

    let mut decoder_data: HashMap<usize, char> = HashMap::new();
    let mut encoder_data: HashMap<char, usize> = HashMap::new();

    for (index, value) in v.iter().enumerate() {
        decoder_data.insert(index, *value);
        encoder_data.insert(*value, index);
    }

    logger.info(&format!("{:?}", encode("hii there", &encoder_data)));
    logger.info(&format!(
        "{:?}",
        decode(encode("hii there", &encoder_data), &decoder_data)
    ));

    Ok(())
}

fn encode<'a>(string: &str, encoder_data: &'a HashMap<char, usize>) -> Vec<&'a usize> {
    string
        .chars()
        .map(|c| encoder_data.get(&c).unwrap())
        .collect::<Vec<&usize>>()
}

fn decode(tokens: Vec<&usize>, decoder_data: &HashMap<usize, char>) -> String {
    tokens
        .into_iter()
        .map(|t| decoder_data.get(t).unwrap())
        .collect::<String>()
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
