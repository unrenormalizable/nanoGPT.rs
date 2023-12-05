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

    //nano_gpt_lib::training::run(&ai, &l)
    type MyBackend = Wgpu<AutoGraphicsApi, f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";
    nano_gpt_lib::training::train::<MyAutodiffBackend>(
        artifact_dir,
        TrainingConfig::new(),
        device.clone(),
        &l,
    );

    Ok(())
}

fn create_args_info<'a>(args: &'a args::NanoGptArgs) -> ArgsInfo<'a> {
    ArgsInfo {
        input_file: args.input_file,
        output_folder: args.output_folder,
    }
}
