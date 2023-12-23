mod args;

use burn_wgpu::WgpuDevice;
use burn::optim::decay::WeightDecayConfig;
use nano_gpt_lib::data::dataset::NanoGptDataset;
use nano_gpt_lib::data::BigramInputsDataset;
use nano_gpt_lib::training::ExperimentConfig;

fn main() {
    type AutoDiffBackend = burn::backend::Autodiff<burn::backend::Wgpu>;

    let config = ExperimentConfig::new(
        burn::optim::AdamConfig::new().with_weight_decay(Some(WeightDecayConfig::new(1.0e-6))),
    );

    nano_gpt_lib::training::train::<AutoDiffBackend, BigramInputsDataset>(
        WgpuDevice::default(),
        //NanoGptDataset::train("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1),
        BigramInputsDataset::new("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1, "D:/delme/bigram-inputs.1.txt"),
        //NanoGptDataset::test("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1),
        BigramInputsDataset::new("D:/src/u/nanoGPT.rs/input.txt", config.block_size + 1, "D:/delme/bigram-inputs.1.txt"),
        config,
        format!("/tmp/text-generation").as_str(),
    );

    // TODO: For inference use NoAutoDiffBackend.
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
