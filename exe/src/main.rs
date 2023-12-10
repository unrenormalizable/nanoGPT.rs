mod args;

fn main() -> Result<(), String> {
    let arg_matches = args::parse_args();
    let args = args::get_args(&arg_matches);
    let ai = create_args_info(&args);

    /*
        let mut rng = rand::thread_rng();
        let indices =
            rand::seq::index::sample(&mut rng, items.len() - self.block_size, self.batch_size)
                .into_vec();
    */

    Ok(())
}

fn create_args_info<'a>(args: &'a args::ArgsInfo) -> args::ArgsInfo<'a> {
    args::ArgsInfo {
        input_file: args.input_file,
        output_folder: args.output_folder,
    }
}
