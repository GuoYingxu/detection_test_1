/// mock data for image buffer 
/// read from  jpg file
use std::{fs::File};

pub fn mock_from_file(path:&str) -> Result<Vec<u8>> {

    Ok(())
}

fn  from_file<P:AsRef<Path>>(path:P) -> Resut<(u32,u32,Arc<Mmap>,Box<dyn std::error::Error>)> {
    let file = File::open(&path);

    Ok((0,0))
}