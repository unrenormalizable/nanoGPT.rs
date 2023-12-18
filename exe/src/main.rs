mod args;

use burn::optim::decay::WeightDecayConfig;
use nano_gpt_lib::data::dataset::NanoGptDataset;
use nano_gpt_lib::training::ExperimentConfig;

type Backend = burn::backend::Autodiff<burn::backend::Wgpu>;

fn main() {
    let config = ExperimentConfig::new(
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    nano_gpt_lib::training::train::<Backend, NanoGptDataset>(
        burn::tensor::Device::<Backend>::default(),
        NanoGptDataset::train("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1),
        NanoGptDataset::test("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1),
        config,
        format!("/tmp/text-generation").as_str(),
    );
}

fn main2() -> Result<(), String> {
    let arg_matches = args::parse_args();
    let args = args::get_args(&arg_matches);
    let ai = create_args_info(&args);

    Ok(())
}

fn create_args_info<'a>(args: &'a args::ArgsInfo) -> args::ArgsInfo<'a> {
    args::ArgsInfo {
        input_file: args.input_file,
        output_folder: args.output_folder,
    }
}
