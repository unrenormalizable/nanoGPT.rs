use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, Int, Tensor},
};

type NanoGptToken = i32;

#[derive(new)]
pub struct NanoGptBatcher<B: Backend> {
    device: B::Device,
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
            .map(Data::from)
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let tokens: Tensor<B, 2, Int> = Tensor::stack(tokens, 0).to_device(&self.device);

        let targets: Vec<Tensor<B, 1, Int>> = indices
            .iter()
            .map(|&i| &items[i + 1..i + self.block_size + 1])
            .map(Data::from)
            .map(|data| Tensor::from_data(data.convert()))
            .collect();
        let targets: Tensor<B, 2, Int> = Tensor::stack(targets, 0).to_device(&self.device);

        NanoGptBatch { tokens, targets }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::*;

    type Backend = NdArray<f32>;

    fn generate_indices(_length: usize, amount: usize) -> Vec<usize> {
        (0 .. amount).collect()
    }

    #[test]
    fn batcher_test() {
        let b: NanoGptBatcher<Backend> = NanoGptBatcher::new(NdArrayDevice::Cpu, 3, 2, generate_indices);

        let x = b.batch(vec![0, 1, 2222, 3, 4, 5]);

        assert_eq!(&x.tokens.to_data(), &Data::from([[0, 1], [1, 2222], [2222, 3]]));
        assert_eq!(&x.targets.to_data(), &Data::from([[1, 2222], [2222, 3], [3, 4]]));
    }
}
