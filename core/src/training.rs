use crate::{
    data::{ CharTokenizer, TextGenerationBatcher, TextGenerationItem, Tokenizer },
    // model::TextGenerationModelConfig,
};
use burn::data::dataset::transform::SamplerDataset;
use burn::{
    config::Config,
    data::{dataloader::DataLoaderBuilder, dataset::Dataset},
    lr_scheduler::noam::NoamLrSchedulerConfig,
    module::Module,
    nn::transformer::TransformerEncoderConfig,
    optim::AdamConfig,
    record::{CompactRecorder, DefaultRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, CUDAMetric, LearningRateMetric, LossMetric},
        LearnerBuilder,
    },
};
use std::sync::Arc;

pub struct ArgsInfo<'a> {
    pub input_file: &'a str,
    pub output_folder: &'a str,
}

#[derive(Config)]
pub struct ExperimentConfig {
    #[config(default = 512)]
    max_seq_length: usize,
    #[config(default = 6)]
    batch_size: usize,
    #[config(default = 50)]
    num_epochs: usize,
}

pub fn train<B: AutodiffBackend, D: Dataset<TextGenerationItem> + 'static>(
    device: B::Device,
    dataset_train: D,
    dataset_test: D,
    config: ExperimentConfig,
    artifact_dir: &str,
) {
    let tokenizer = Arc::new(CharTokenizer::default());
    let batcher_train = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);
    let batcher_test = TextGenerationBatcher::new(tokenizer.clone(), config.max_seq_length);

    //let model = TextGenerationModelConfig::new(
    //    config.transformer.clone(),
    //    tokenizer.vocab_size(),
    //    tokenizer.pad_token(),
    //    config.max_seq_length,
    //)
    //.init::<B>();

    //let dataloader_train = DataLoaderBuilder::new(batcher_train)
    //    .batch_size(config.batch_size)
    //    .num_workers(4)
    //    .build(SamplerDataset::new(dataset_train, 10_000));

    //let dataloader_test = DataLoaderBuilder::new(batcher_test)
    //    .batch_size(config.batch_size)
    //    .num_workers(4)
    //    .build(SamplerDataset::new(dataset_test, 1000));

    //let accum = 6; // Effective batch size = 6 * 6 = 32.
    //let optim = config.optimizer.init();
    //let lr_scheduler = NoamLrSchedulerConfig::new(0.01 / accum as f64)
    //    .with_warmup_steps(6000)
    //    .with_model_size(config.transformer.d_model)
    //    .init();

    //let learner = LearnerBuilder::new(artifact_dir)
    //    .metric_train(CUDAMetric::new())
    //    .metric_valid(CUDAMetric::new())
    //    .metric_train_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
    //    .metric_valid_numeric(AccuracyMetric::new().with_pad_token(tokenizer.pad_token()))
    //    .metric_train(LossMetric::new())
    //    .metric_valid(LossMetric::new())
    //    .metric_train_numeric(LearningRateMetric::new())
    //    .with_file_checkpointer(CompactRecorder::new())
    //    .devices(vec![device])
    //    .grads_accumulation(accum)
    //    .num_epochs(config.num_epochs)
    //    .build(model, optim, lr_scheduler);

    //let model_trained = learner.fit(dataloader_train, dataloader_test);

    //config.save(format!("{artifact_dir}/config.json")).unwrap();

    //DefaultRecorder::new()
    //    .record(
    //        model_trained.into_record(),
    //        format!("{artifact_dir}/model").into(),
    //    )
    //    .unwrap();
}
