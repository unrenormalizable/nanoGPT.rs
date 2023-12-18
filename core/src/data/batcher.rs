use crate::{data::dataset::NanoGptItem, data::tokenizer::Tokenizer};
use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Int, Shape, Tensor},
};
use std::sync::Arc;

#[derive(new)]
pub struct NanoGptBatcher {
    tokenizer: Arc<dyn Tokenizer>,
}

#[derive(Clone, Debug)]
pub struct NanoGptBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<NanoGptItem, NanoGptBatch<B>> for NanoGptBatcher {
    fn batch(&self, items: Vec<NanoGptItem>) -> NanoGptBatch<B> {
        let tokens: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|x| x.chars[..x.chars.len() - 1].to_vec())
            .map(|x| (x.len(), self.tokenizer.encode(&x)))
            .map(|(l, x)| Data::new(x, Shape::new([l])))
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let tokens: Tensor<B, 2, Int> = Tensor::stack(tokens, 0).to_device(&B::Device::default());

        let targets: Vec<Tensor<B, 1, Int>> = items
            .iter()
            .map(|x| x.chars[1..].to_vec())
            .map(|x| (x.len(), self.tokenizer.encode(&x)))
            .map(|(l, x)| Data::new(x, Shape::new([l])))
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let targets: Tensor<B, 2, Int> = Tensor::stack(targets, 0).to_device(&B::Device::default());

        NanoGptBatch { tokens, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::tokenizer::CharTokenizer;
    use burn_ndarray::*;

    type Backend = NdArray<f32>;

    #[test]
    fn batcher_test() {
        let b: NanoGptBatcher = NanoGptBatcher::new(Arc::new(CharTokenizer::default()));

        let x: NanoGptBatch<Backend> = b.batch(vec![
            NanoGptItem::new(vec!['h', 'e', 'l', 'l', 'o', '?']),
            NanoGptItem::new(vec!['w', 'o', 'r', 'l', 'd', '!']),
        ]);

        assert_eq!(
            &x.tokens.to_data(),
            &Data::from([[46, 43, 50, 50, 53], [61, 53, 56, 50, 42]])
        );
        assert_eq!(
            &x.targets.to_data(),
            &Data::from([[43, 50, 50, 53, 12], [53, 56, 50, 42, 2]])
        );
    }
}
