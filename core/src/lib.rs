pub mod logger;

use crate::logger::NanoGptLogger;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;

pub struct ArgsInfo<'a> {
    pub input_file: &'a str,
    pub output_folder: &'a str,
}

pub fn run(args_info: &ArgsInfo, logger: &dyn NanoGptLogger) -> Result<(), String> {
    logger.info(&format!(
        "Starting with {:?},  {:?}.",
        args_info.input_file, args_info.output_folder
    ));

    let mut file = File::open(args_info.input_file).map_err(|e| e.to_string())?;
    let mut contents = String::new();

    file.read_to_string(&mut contents)
        .map_err(|e| e.to_string())?;

    logger.info(&format!("# of characters {:?}", contents.len()));
    let s: std::collections::HashSet<char> = contents.chars().collect();
    let mut v = Vec::from_iter(s);
    v.sort();

    let mut decoder_data: HashMap<usize, char> = HashMap::new();
    let mut encoder_data: HashMap<char, usize> = HashMap::new();

    for (index, value) in v.iter().enumerate() {
        decoder_data.insert(index, *value);
        encoder_data.insert(*value, index);
    }

    //logger.info(&format!("# of characters {:?} => {:?}", v.len(), v));
    logger.info(&format!("{:?}", encode("hii there", &encoder_data)));
    logger.info(&format!(
        "{:?}",
        decode(encode("hii there", &encoder_data), &decoder_data)
    ));

    Ok(())
}

fn encode<'a>(string: &str, encoder_data: &'a HashMap<char, usize>) -> Vec<&'a usize> {
    string
        .chars()
        .map(|c| encoder_data.get(&c).unwrap())
        .collect::<Vec<&usize>>()
}

fn decode(tokens: Vec<&usize>, decoder_data: &HashMap<usize, char>) -> String {
    tokens
        .into_iter()
        .map(|t| decoder_data.get(t).unwrap())
        .collect::<String>()
}

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
