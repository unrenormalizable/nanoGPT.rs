pub trait NanoGptLogger {
    fn info(&self, msg: &str);
    fn error(&self, msg: &str);
    fn warning(&self, msg: &str);
}
