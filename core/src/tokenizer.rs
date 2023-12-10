use std::collections::HashMap;

pub trait Tokenizer: Send + Sync {
    fn encode(&self, chars: &[char]) -> Vec<i32>;
    fn decode(&self, tokens: &[i32]) -> Vec<char>;
    fn vocab_size(&self) -> i32;
}

pub struct CharTokenizer {
    decoder_data: HashMap<i32, char>,
    encoder_data: HashMap<char, i32>,
    chars: String,
}

impl Default for CharTokenizer {
    fn default() -> Self {
        // NOTE: Hard coding it here. Eventually generate it outside code & download it here.
        let chars =
            String::from("\n!$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz");
        let mut decoder_data: HashMap<i32, char> = HashMap::new();
        let mut encoder_data: HashMap<char, i32> = HashMap::new();

        for (index, value) in chars.chars().enumerate() {
            decoder_data.insert(index as i32, value);
            encoder_data.insert(value, index as i32);
        }

        Self {
            chars,
            decoder_data,
            encoder_data,
        }
    }
}

impl Tokenizer for CharTokenizer {
    fn encode(&self, chars: &[char]) -> Vec<i32> {
        chars
            .iter()
            .map(|&c| *self.encoder_data.get(&c).unwrap())
            .collect()
    }

    fn decode(&self, tokens: &[i32]) -> Vec<char> {
        tokens
            .iter()
            .map(|t| *self.decoder_data.get(t).unwrap())
            .collect()
    }

    fn vocab_size(&self) -> i32 {
        self.chars.len() as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() {
        let tokenizer = CharTokenizer::default();
        let chars = tokenizer.chars.chars().collect::<Vec<char>>();

        let tokens = tokenizer.encode(&chars);
        let decoded = tokenizer.decode(&tokens);

        assert_eq!(decoded, chars);
    }
}
