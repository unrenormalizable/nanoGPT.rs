use burn::data::dataset::Dataset;
use std::fs;

pub type NanoGptItem = char;

pub struct NanoGptDataset {
    dataset: String,
}

impl Dataset<NanoGptItem> for NanoGptDataset {
    fn get(&self, index: usize) -> Option<NanoGptItem> {
        self.dataset
            .chars()
            .nth(index)
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl NanoGptDataset {
    pub fn train(data_file: &str) -> Self {
        Self::new(data_file, "train")
    }

    pub fn test(data_file: &str) -> Self {
        Self::new(data_file, "test")
    }

    pub fn new(data_file: &str, split: &str) -> Self {
        let contents = fs::read_to_string(data_file).unwrap();
        let (train, test) = contents.split_at(contents.len() * 9 / 10);

        let dataset = match split {
            "train" => train,
            "test" => test,
            _ => panic!("Asked for unknown dataset {}...", split),
        };

        Self { dataset: String::from(dataset) }
    }
}
