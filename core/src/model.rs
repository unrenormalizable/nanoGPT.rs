use crate::data::*;
use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLoss, Embedding, EmbeddingConfig},
    tensor::backend::{AutodiffBackend, Backend},
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

#[derive(Module, Debug)]
pub struct NanoGptModel<B: Backend> {
    token_embedding_table: Embedding<B>,
}

#[derive(Config, Debug)]
pub struct NanoGptModelConfig {
    vocab_size: usize,
}

impl NanoGptModelConfig {
    pub fn init<B: Backend>(&self) -> NanoGptModel<B> {
        let token_embedding_table = EmbeddingConfig::new(self.vocab_size, self.vocab_size)
        .with_initializer(burn::nn::Initializer::Ones)
        .init(); // TODO: Explicitly initialize with device with next burn update.

        NanoGptModel {
            token_embedding_table,
        }
    }
}

impl<B: Backend> NanoGptModel<B> {
    pub fn forward_training(&self, item: NanoGptBatch<B>) -> ClassificationOutput<B> {
        let tokens = item.tokens;
        let targets = item.targets;

        println!("item.tokens = {}", tokens);
        println!("item.targets = {}", targets);
        let logits = self.token_embedding_table.forward(tokens.clone());

        let l_s = logits.shape();
        let logits = logits.reshape([l_s.dims[0] * l_s.dims[1], l_s.dims[2]]);

        let t_s = targets.shape();
        let targets = targets.clone().reshape([t_s.dims[0] * t_s.dims[1]]);

        let loss = CrossEntropyLoss::default().forward(logits.clone(), targets.clone());
        println!("loss = {}", loss);

        ClassificationOutput {
            loss,
            output: logits,
            targets,
        }
    }
}

impl<B: AutodiffBackend> TrainStep<NanoGptBatch<B>, ClassificationOutput<B>> for NanoGptModel<B> {
    fn step(&self, item: NanoGptBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_training(item);
        let grads = item.loss.backward();

        TrainOutput::new(self, grads, item)
    }
}

impl<B: Backend> ValidStep<NanoGptBatch<B>, ClassificationOutput<B>> for NanoGptModel<B> {
    fn step(&self, item: NanoGptBatch<B>) -> ClassificationOutput<B> {
        self.forward_training(item)
    }
}
