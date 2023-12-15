use burn::{
    config::Config,
    module::Module,
    nn::{loss::CrossEntropyLoss, Embedding, EmbeddingConfig},
    tensor::{backend::Backend, Int, Shape, Tensor},
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
        let token_embedding_table = EmbeddingConfig::new(self.vocab_size, self.vocab_size).init();

        NanoGptModel {
            token_embedding_table,
        }
    }
}

impl<B: Backend> NanoGptModel<B> {
    pub fn forward(
        &self,
        idx: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 2>, Tensor<B, 1>) {
        let logits = self.token_embedding_table.forward(idx);

        let l_s = logits.shape();
        let logits = logits.reshape(Shape::new([l_s.dims[0] * l_s.dims[1], l_s.dims[2]]));

        let t_s = targets.shape();
        let targets = targets.reshape(Shape::new([t_s.dims[0] * t_s.dims[1]]));

        let loss = CrossEntropyLoss::default().forward(logits.clone(), targets);

        (logits, loss)
    }
}
