use ort::{ep::CUDA, session::Session};
use anyhow::{Result};
pub struct RoiSession {
    session:Session
}

impl RoiSession {

    pub fn create_session()->Result<Self> {
        
        let s = Session::builder()?
        .with_execution_providers([
            CUDA::default().build()
        ])?
        .commit_from_file("unet.onnx")?;
        Ok(RoiSession {session: s})
    }


}
