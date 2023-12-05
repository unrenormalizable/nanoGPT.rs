// TODO: Kill this in favor of log crate macros.
pub trait NanoGptLogger {
    fn info(&self, msg: &str);
    fn error(&self, msg: &str);
    fn warning(&self, msg: &str);
}
