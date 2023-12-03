use chrono::Local;
use colored::*;
use nano_gpt_lib::logger::NanoGptLogger;

fn get_time_stamp() -> String {
    Local::now().format("%y-%m-%d %H:%M:%S").to_string()
}

pub(crate) struct ColoredConsoleLogger;

impl NanoGptLogger for ColoredConsoleLogger {
    fn info(&self, msg: &str) {
        println!(
            "{} {}",
            get_time_stamp().white(),
            format!("info: {}", msg).green(),
        );
    }

    fn error(&self, msg: &str) {
        println!(
            "{} {}",
            get_time_stamp().white(),
            format!("error: {}", msg).red(),
        );
    }

    fn warning(&self, msg: &str) {
        println!(
            "{} {}",
            get_time_stamp().white(),
            format!("warning: {}", msg).yellow(),
        );
    }
}
