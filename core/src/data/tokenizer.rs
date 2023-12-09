use std::collections::HashMap;

pub trait Tokenizer: Send + Sync {
    fn encode_one(&self, value: &char) -> usize;
    fn encode(&self, value: &str) -> Vec<usize>;
    fn decode_one(&self, tokens: &usize) -> char;
    fn decode(&self, tokens: &[usize]) -> String;
    fn vocab_size(&self) -> usize;
}

pub struct CharTokenizer {
    decoder_data: HashMap<usize, char>,
    encoder_data: HashMap<char, usize>,
    chars: String,
}

impl Default for CharTokenizer {
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

impl Tokenizer for CharTokenizer {
    fn encode_one(&self, value: &char) -> usize {
        *self.encoder_data.get(&value).unwrap()
    }

    fn encode(&self, value: &str) -> Vec<usize> {
        value
            .chars()
            .map(|c| self.encode_one(&c))
            .collect::<Vec<usize>>()
    }

    fn decode_one(&self, token: &usize) -> char {
        *self.decoder_data.get(token).unwrap()
    }

    fn decode(&self, tokens: &[usize]) -> String {
        tokens
            .into_iter()
            .map(|t| self.decode_one(t))
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
        let tokenizer = CharTokenizer::default();
        let text = tokenizer.chars.clone();

        let tokens = tokenizer.encode(&text);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, text);
    }
}
