use std::time::Instant;

use ort::{ep::CUDA, session::{Session},value::TensorRef,inputs};
use anyhow::{Result};
use ndarray::{Array4, ArrayView4};
pub struct RoiSession {
    session:Session,
    data:Vec<f32>
}

impl RoiSession {

    pub fn create_session()->Result<Self> {
        
        let s = Session::builder()?
        .with_execution_providers([
            CUDA::default().build()
        ])?
        .commit_from_file("unet.onnx")?;
        Ok(RoiSession {session: s,data:Vec::<f32>::with_capacity(1024*1024)})
    }

    pub fn run_with_vec(&mut self, data:&Vec<f32>)->Result<()> {
        let start = Instant::now();
        let tensor = ArrayView4::<f32>::from_shape(
            (1, 1, 1024, 1024),
            data
        )?;
        let output = self.session.run(inputs![TensorRef::from_array_view(tensor)?])?;
        println!("输出张量shape{},耗时{}ms",output[0].shape(),start.elapsed().as_millis());
        Ok(())
    }

}
