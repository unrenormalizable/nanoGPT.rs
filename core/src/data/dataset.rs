use burn::data::dataset::{ Dataset, InMemDataset };
use std::fs::File;
use std::io::Read;

pub type TextGenerationItem = char;

type ShakespeareItem = char;

pub struct ShakespeareDataset {
    dataset: InMemDataset<ShakespeareItem>,
}

impl Dataset<TextGenerationItem> for ShakespeareDataset {
    fn get(&self, index: usize) -> Option<TextGenerationItem> {
        self.dataset.get(index)
            
    }

    fn len(&self) -> usize {
        self.dataset.len()
    }
}

impl ShakespeareDataset {
    pub fn train(file_name: &str) -> Self {
        Self::new(file_name, "train")
    }

    pub fn test(file_name: &str) -> Self {
        Self::new(file_name, "test")
    }

    pub fn new(file_name: &str, split: &str) -> Self {
        let mut file = File::open(file_name).unwrap();
        let mut contents = String::new();

        file.read_to_string(&mut contents).unwrap();

        let dataset: InMemDataset<ShakespeareItem> = InMemDataset::new(contents.chars().collect());
        Self { dataset }
    }
}
