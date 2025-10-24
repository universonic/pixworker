use anyhow::{Ok, Result};
use pixworker::cmd::root::{RootCmd};

fn main() -> Result<()> {
    RootCmd::new().run()?;
    Ok(())
}
