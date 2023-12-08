use burn::{
    data::dataloader::batcher::Batcher,
    tensor::{backend::Backend, Data, ElementConversion, Int, Tensor},
};
use burn_dataset::{
    transform::{Mapper, MapperDataset},
    Dataset, InMemDataset,
};
use serde::{Deserialize, Serialize};

const WIDTH: usize = 28;
const HEIGHT: usize = 28;

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct NanoGptItem {
    pub image: [[f32; WIDTH]; HEIGHT],

    pub label: usize,
}

#[derive(Deserialize, Debug, Clone)]
struct NanoGptItemRaw {
    pub image_bytes: Vec<u8>,

    pub label: usize,
}

struct BytesToImage;

impl Mapper<NanoGptItemRaw, NanoGptItem> for BytesToImage {
    /// Convert a raw NanoGpt item (image bytes) to a NanoGpt item (2D array image).
    fn map(&self, item: &NanoGptItemRaw) -> NanoGptItem {
        let image = [[0 as u8; WIDTH]; HEIGHT];

        // Ensure the image dimensions are correct.
        debug_assert_eq!((image.len(), image[0].len()), (WIDTH, HEIGHT));

        // Convert the image to a 2D array of floats.
        let mut image_array = [[0f32; WIDTH]; HEIGHT];
        NanoGptItem {
            image: image_array,
            label: item.label,
        }
    }
}

type MappedDataset = MapperDataset<InMemDataset<NanoGptItemRaw>, BytesToImage, NanoGptItemRaw>;

/// NanoGpt dataset from Huggingface.
///
/// The data is downloaded from Huggingface and stored in a SQLite database.
pub struct NanoGptDataset {
    dataset: MappedDataset,
}

impl Dataset<NanoGptItem> for NanoGptDataset {
    fn get(&self, index: usize) -> Option<NanoGptItem> {
        self.dataset.get(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl NanoGptDataset {
    /// Creates a new train dataset.
    pub fn train() -> Self {
        Self::new("train")
    }

    /// Creates a new test dataset.
    pub fn test() -> Self {
        Self::new("test")
    }

    fn new(split: &str) -> Self {
        // TOOD: Switch to delayed file loader
        let dataset = InMemDataset::from_json_rows("mnist")
            // Use: split here.
            .unwrap();

        let dataset = MapperDataset::new(dataset, BytesToImage);

        Self { dataset }
    }
}

pub struct NanoGptBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> NanoGptBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct NanoGptBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<NanoGptItem, NanoGptBatch<B>> for NanoGptBatcher<B> {
    fn batch(&self, items: Vec<NanoGptItem>) -> NanoGptBatch<B> {
        let images = items
            .iter()
            .map(|item| Data::<f32, 2>::from(item.image))
            .map(|data| Tensor::<B, 2>::from_data(data.convert()))
            .map(|tensor| tensor.reshape([1, 28, 28]))
            .map(|tensor| ((tensor / 255) - 0.1307) / 0.3081)
            .collect();

        let targets = items
            .iter()
            .map(|item| Tensor::<B, 1, Int>::from_data([(item.label as i64).elem()]))
            .collect();

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        NanoGptBatch { images, targets }
    }
}
