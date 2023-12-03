mod args;
mod logger;

use nano_gpt_lib::*;

fn main() -> Result<(), String> {
    let l = logger::ColoredConsoleLogger {};

    let arg_matches = args::parse_args();
    let args = args::get_args(&arg_matches);

    let ai = create_args_info(&args);

    nano_gpt_lib::run(&ai, &l)
}

fn create_args_info<'a>(args: &'a args::NanoGptArgs) -> ArgsInfo<'a> {
    ArgsInfo {
        input_file: args.input_file,
        output_folder: args.output_folder,
    }
}
