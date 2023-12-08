use std::collections::HashMap;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, value: &str) -> Vec<usize>;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

pub struct NanoGptTokenizer {
    decoder_data: HashMap<usize, char>,
    encoder_data: HashMap<char, usize>,
    chars: String,
}

impl Default for NanoGptTokenizer {
    fn default() -> Self {
        // NOTE: Hard coding it here. Eventually generate it outside code & download it here.
        let chars =
            String::from("\n!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        let mut decoder_data: HashMap<usize, char> = HashMap::new();
        let mut encoder_data: HashMap<char, usize> = HashMap::new();

        for (index, value) in chars.chars().enumerate() {
            decoder_data.insert(index, value);
            encoder_data.insert(value, index);
        }

        Self {
            chars,
            decoder_data,
            encoder_data,
        }
    }
}

impl Tokenizer for NanoGptTokenizer {
    fn encode(&self, value: &str) -> Vec<usize> {
        value
            .chars()
            .map(|c| *self.encoder_data.get(&c).unwrap())
            .collect::<Vec<usize>>()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .into_iter()
            .map(|t| self.decoder_data.get(t).unwrap())
            .collect::<String>()
    }

    fn vocab_size(&self) -> usize {
        self.chars.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = NanoGptTokenizer::default();
        let text = tokenizer.chars.clone();

        let tokens = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }
}
