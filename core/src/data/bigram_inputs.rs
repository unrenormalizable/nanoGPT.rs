use crate::data::NanoGptItem;
use burn::data::dataset::Dataset;
use std::fs::read_to_string;

pub struct BigramInputsDataset {
    dataset: String,
    max_seq_length: usize,
    indices: Vec<usize>,
}

impl BigramInputsDataset  {
    pub fn new(data_file: &str, max_seq_length: usize, index_file: &str) -> Self {
        let mut indices = Vec::new();

        for line in read_to_string(index_file).unwrap().lines().filter(|&x| x.trim().len() != 0) {
            indices.push(line.split('\0').collect::<Vec<&str>>()[0].parse::<usize>().unwrap())
        }

        Self {
            dataset: read_to_string(data_file).unwrap(),
            max_seq_length,
            indices,
        }
    }
}

impl Dataset<NanoGptItem> for BigramInputsDataset {
    fn get(&self, index: usize) -> Option<NanoGptItem> {
        if index >= self.indices.len() {
            return None;
        }

        Some(NanoGptItem::new(
            index,
            self.dataset
                .chars()
                .skip(self.indices[index])
                .take(self.max_seq_length)
                .collect::<Vec<char>>(),
        ))
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.max_seq_length + 1
    }
}
