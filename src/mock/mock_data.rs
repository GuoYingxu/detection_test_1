/// mock data for image buffer 
/// read from  jpg file
use std::{fs::File};
use anyhow::Context;
pub struct MockData {
    width: u32,
    height: u32,
    data: Vec<u8>
}

impl MockData {
    fn from_jpg(&self)-> self {

    }
}

