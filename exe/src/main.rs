use burn::backend::wgpu::AutoGraphicsApi;
use burn::backend::{Autodiff, Wgpu};
mod args;
mod logger;

use nano_gpt_lib::training::*;

fn main() -> Result<(), String> {
    let l = logger::ColoredConsoleLogger {};

    let arg_matches = args::parse_args();
    let args = args::get_args(&arg_matches);

    let ai = create_args_info(&args);

    let config = ExperimentConfig::new();

    text_generation::training::train::<Backend, DbPediaDataset>(
        burn::tensor::Device::<Backend>::Cuda(0),
        DbPediaDataset::train(),
        DbPediaDataset::test(),
        config,
        "/tmp/text-generation",
    );

    Ok(())
}

fn create_args_info<'a>(args: &'a args::NanoGptArgs) -> ArgsInfo<'a> {
    ArgsInfo {
        input_file: args.input_file,
        output_folder: args.output_folder,
    }
}
