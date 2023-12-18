use burn::data::dataset::Dataset;
use std::fs;

#[derive(new, Clone, Debug)]
pub struct NanoGptItem {
    pub chars: Vec<char>,
}

pub struct NanoGptDataset {
    dataset: String,
    max_seq_length: usize,
}

impl Dataset<NanoGptItem> for NanoGptDataset {
    fn get(&self, index: usize) -> Option<NanoGptItem> {
        if index >= self.len() {
            return None;
        }

        Some(NanoGptItem::new(
            self.dataset
                .chars()
                .skip(index)
                .take(self.max_seq_length)
                .collect::<Vec<char>>(),
        ))
    }

    fn len(&self) -> usize {
        self.dataset.len() - self.max_seq_length + 1
    }
}

impl NanoGptDataset {
    pub fn train(data_file: &str, batch_size: usize) -> Self {
        Self::new(data_file, "train", batch_size)
    }

    pub fn test(data_file: &str, batch_size: usize) -> Self {
        Self::new(data_file, "test", batch_size)
    }

    pub fn new(data_file: &str, split: &str, batch_size: usize) -> Self {
        let contents = fs::read_to_string(data_file).unwrap();
        Self::new_with_contents(&contents, split, batch_size)
    }

    pub fn new_with_contents(contents: &str, split: &str, batch_size: usize) -> Self {
        let (train, test) = contents.split_at(contents.len() * 9 / 10);

        let dataset = match split {
            "train" => train,
            "test" => test,
            _ => panic!("Asked for unknown dataset {}...", split),
        };

        Self {
            dataset: String::from(dataset),
            max_seq_length: batch_size + 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    pub fn check_length() {
        let ds = NanoGptDataset::new_with_contents("0123456789", "train", 4);

        assert_eq!(5, ds.len());
    }

    #[test]
    pub fn check_get_1() {
        let ds = NanoGptDataset::new_with_contents("0123456789", "train", 4);

        assert_eq!(
            Some(vec!['0', '1', '2', '3', '4']),
            ds.get(0).map(|x| x.chars)
        );
    }

    #[test]
    pub fn check_get_edge_case_1() {
        let ds = NanoGptDataset::new_with_contents("0123456789", "train", 4);

        assert_eq!(
            Some(vec!['4', '5', '6', '7', '8']),
            ds.get(4).map(|x| x.chars)
        );
    }

    #[test]
    pub fn check_get_edge_case_2() {
        let ds = NanoGptDataset::new_with_contents("0123456789", "train", 4);

        assert_eq!(None, ds.get(5).map(|x| x.chars));
    }
}
