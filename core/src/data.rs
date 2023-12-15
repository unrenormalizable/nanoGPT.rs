use crate::tokenizer::Tokenizer;
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Int, Shape, Tensor},
};
use std::sync::Arc;

type NanoGptToken = char;

#[derive(new)]
pub struct NanoGptBatcher<B: Backend> {
    device: B::Device,
    tokenizer: Arc<dyn Tokenizer>,
    batch_size: usize,
    block_size: usize,
    generate_indices: fn(length: usize, amount: usize) -> Vec<usize>
}

#[derive(Clone, Debug)]
pub struct NanoGptBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<NanoGptToken, NanoGptBatch<B>> for NanoGptBatcher<B> {
    fn batch(&self, items: Vec<NanoGptToken>) -> NanoGptBatch<B> {
        let indices = (self.generate_indices)(items.len() - self.block_size, self.batch_size);

        let tokens: Vec<Tensor<B, 1, Int>> = indices
            .iter()
            .map(|&i| &items[i..i + self.block_size])
            .map(|x| self.tokenizer.encode(x))
            .map(|x| Data::new(x, Shape::new([self.block_size])))
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let tokens: Tensor<B, 2, Int> = Tensor::stack(tokens, 0).to_device(&self.device);

        let targets: Vec<Tensor<B, 1, Int>> = indices
            .iter()
            .map(|&i| &items[i + 1..i + self.block_size + 1])
            .map(|x| self.tokenizer.encode(x))
            .map(|x| Data::new(x, Shape::new([self.block_size])))
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let targets: Tensor<B, 2, Int> = Tensor::stack(targets, 0).to_device(&self.device);

        NanoGptBatch { tokens, targets }
    }
}

#[cfg(test)]
mod tests {
    use crate::tokenizer::CharTokenizer;
    use super::*;
    use burn_ndarray::*;

    type Backend = NdArray<f32>;

    fn generate_indices(_length: usize, amount: usize) -> Vec<usize> {
        (0 .. amount).collect()
    }

    #[test]
    fn batcher_test() {
        let b: NanoGptBatcher<Backend> = NanoGptBatcher::new(NdArrayDevice::Cpu, Arc::new(CharTokenizer::default()), 3, 2, generate_indices);

        let x = b.batch("hello?".chars().collect());

        assert_eq!(&x.tokens.to_data(), &Data::from([[45, 42], [42, 49], [49, 49]]));
        assert_eq!(&x.targets.to_data(), &Data::from([[42, 49], [49, 49], [49, 52]]));
    }
}
