use burn::data::dataset::Dataset;
use std::marker::PhantomData;

pub struct RangeSamplerDataset<D, I, O> {
    dataset: D,
    block_size: usize,
    input: PhantomData<I>,
    output: PhantomData<O>,
}

impl<D, I, O> RangeSamplerDataset<D, I, O>
where
    D: Dataset<I>,
    I: Send + Sync,
{
    pub fn new(dataset: D, block_size: usize) -> Self {
        Self {
            dataset,
            block_size,
            input: PhantomData,
            output: PhantomData,
        }
    }

    ///// Creates a new sampler dataset with replacement.
    //pub fn with_replacement(dataset: D, size: usize) -> Self {
    //    Self::new(dataset, size)
    //}

    ///// Creates a new sampler dataset without replacement.
    //pub fn without_replacement(dataset: D, size: usize) -> Self {
    //    Self {
    //        dataset,
    //        size,
    //        state: Mutex::new(SamplerState::WithoutReplacement(
    //            StdRng::from_entropy(),
    //            Vec::new(),
    //        )),
    //        input: PhantomData,
    //    }
    //}

    //fn index(&self) -> usize {
    //    let mut state = self.state.lock().unwrap();

    //    match state.deref_mut() {
    //        SamplerState::WithReplacement(rng) => rng.sample(Uniform::new(0, self.dataset.len())),
    //        SamplerState::WithoutReplacement(rng, indices) => {
    //            if indices.is_empty() {
    //                // Refill the state.
    //                *indices = (0..self.dataset.len()).choose_multiple(rng, self.dataset.len());
    //            }

    //            indices.pop().expect("Indices are refilled when empty.")
    //        }
    //    }
    //}
}

impl<D, I, O> Dataset<O> for RangeSamplerDataset<D, I, O>
where
    D: Dataset<I>,
    I: Send + Sync,
    O: Send + Sync,
{
    fn get(&self, index: usize) -> Option<O> {
        //if index >= self.dataset.len() + self.block_size {
        return None;
        //}

        //self.dataset.get(self.index())
        //vec![self.dataset.get(1).unwrap()].into_iter().collect::<Option<Vec<I>>>();
    }

    fn len(&self) -> usize {
        0
    }
}
