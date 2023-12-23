use crate::{
    data::{CharTokenizer, NanoGptBatcher, NanoGptItem, Tokenizer},
    model::NanoGptModelConfig,
};
use burn::{
    config::Config,
    data::{
        dataloader::DataLoaderBuilder,
        dataset::{transform::SamplerDataset, Dataset},
    },
    lr_scheduler::{noam::NoamLrSchedulerConfig, constant::ConstantLr},
    module::Module,
    optim::AdamConfig,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

#[derive(Config)]
pub struct ExperimentConfig {
    pub optimizer: AdamConfig,
    #[config(default = 32)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub block_size: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 1)]
    pub num_workers: usize,
    #[config(default = 30)]
    pub num_epochs: usize,
}

// TODO: Pass the device, dont use ::Default()
pub fn train<B: AutodiffBackend, D: Dataset<NanoGptItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(CharTokenizer::default());
    let batcher_train = NanoGptBatcher::<B>::new(device.clone(), tokenizer.clone());
    let batcher_test = NanoGptBatcher::<B::InnerBackend>::new(device.clone(), tokenizer.clone());

    let model = NanoGptModelConfig::new(tokenizer.vocab_size()).init::<B>();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers)
        //.shuffle(config.seed)  // TODO: pass config.seed to samplerdataset
        .build(SamplerDataset::new(dataset_train, 96000 / config.batch_size / config.num_epochs)); // TOOD: what should this be?
        //.build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        .batch_size(config.batch_size)
        .num_workers(config.num_workers) 
        //.shuffle(config.seed) // TODO: pass config.seed to samplerdataset
        .build(SamplerDataset::new(dataset_test, 96000 / config.batch_size / config.num_epochs)); // TOOD: what should this be?
        //.build(dataset_test);

    let optim = config.optimizer.init();
    let lr_scheduler = ConstantLr::new(1e-2);

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .num_epochs(config.num_epochs)
        .build(model, optim, lr_scheduler);

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    config.save(format!("{artifact_dir}/config.json")).unwrap();

    DefaultRecorder::new()
        .record(
            model_trained.into_record(),
            format!("{artifact_dir}/model").into(),
        )
        .unwrap();
}
