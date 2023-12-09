use clap::{App, Arg, ArgMatches};
use std::path::Path;

pub struct ArgsInfo<'a> {
    pub input_file: &'a str,
    pub output_folder: &'a str,
}

pub(crate) fn get_args<'a>(args: &'a ArgMatches) -> ArgsInfo<'a> {
    ArgsInfo {
        input_file: args
            .value_of("INPUT_FILE")
            .expect("This is a required argument"),
        output_folder: args
            .value_of("OUTPUT_FOLDER")
            .expect("This is a required argument"),
    }
}

pub(crate) fn parse_args<'a>() -> ArgMatches<'a> {
    App::new(env!("CARGO_PKG_NAME"))
        .version(env!("CARGO_PKG_VERSION"))
        .author(env!("CARGO_PKG_AUTHORS"))
        .about(env!("CARGO_PKG_DESCRIPTION"))
        .arg(create_input_file_arg())
        .arg(create_output_folder_arg())
        .get_matches()
}

fn create_output_folder_arg<'a, 'b>() -> Arg<'a, 'b> {
    Arg::with_name("OUTPUT_FOLDER")
        .short("o")
        .long("output-folder")
        .value_name("OUTPUT_FOLDER")
        .help("The output directory (will be created if it does not exist).")
        .required(false)
        .takes_value(true)
}

fn create_input_file_arg<'a, 'b>() -> Arg<'a, 'b> {
    Arg::with_name("INPUT_FILE")
        .short("i")
        .long("input")
        .value_name("INPUT_FILE")
        .help("Input file with all the text.")
        .required(true)
        .validator(|s| validate_file_exists(&s))
        .takes_value(true)
}

fn validate_file_exists(s: &str) -> Result<(), String> {
    if Path::new(&s).is_file() {
        Ok(())
    } else {
        Err(format!("'{}' does not exist.", s))
    }
}
