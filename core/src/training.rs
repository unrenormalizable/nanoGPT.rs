use crate::{
    data::{CharTokenizer, NanoGptBatcher, NanoGptItem, Tokenizer},
    model::NanoGptModelConfig,
};
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
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
    optimizer: AdamConfig,
    #[config(default = 4)]
    batch_size: usize,
    #[config(default = 8)]
    block_size: usize,
    #[config(default = 1337)]
    pub seed: u64,
    #[config(default = 50)]
    num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<NanoGptItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(CharTokenizer::default());
    let batcher_train = NanoGptBatcher::new(tokenizer.clone(), config.batch_size, config.block_size, generate_indices);
    let batcher_test = NanoGptBatcher::new(tokenizer.clone(), config.batch_size, config.block_size, generate_indices);

    let model = NanoGptModelConfig::new(
        tokenizer.vocab_size(),
    )
    .init::<B>();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        //.batch_size(config.block_size)    // TODO: Reconcile batch size with Karpathy's code.
        .batch_size(100)
        .num_workers(config.batch_size)
        .shuffle(config.seed)
        .build(dataset_train);

    let dataloader_test = DataLoaderBuilder::new(batcher_test)
        //.batch_size(config.block_size)    // TODO: Reconcile batch size with Karpathy's code.
        .batch_size(100)
        .num_workers(config.batch_size)
        .shuffle(config.seed)
        .build(dataset_test);

    let accum = 6; // Effective batch size = 6 * 6 = 32.
    let optim = config.optimizer.init();
    let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
        .with_warmup_steps(6000)
        // .with_model_size(config.transformer.d_model): TODO: what is this?
        .init();

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train(CUDAMetric::new())
        .metric_valid(CUDAMetric::new())
        // TODO: what is this?
        //.metric_train_numeric(AccuracyMetric::new())
        //.metric_valid_numeric(AccuracyMetric::new())
        .metric_train(LossMetric::new())
        .metric_valid(LossMetric::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device])
        .grads_accumulation(accum)
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

fn generate_indices(length: usize, amount: usize) -> Vec<usize> {
    let mut rng = rand::thread_rng();
    let indices =
        rand::seq::index::sample(&mut rng, length, amount)
            .into_vec();

    indices
}