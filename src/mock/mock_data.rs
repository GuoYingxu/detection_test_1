use anyhow::{Context, Result};
use image::{ImageError, ImageReader};
use memmap2::Mmap;
/// mock data for image buffer
/// read from  jpg file
use std::{fs::File, io::Cursor, ptr::read};
pub struct MockData {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl MockData {
    pub fn from_jpg(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let file = File::open(path).with_context(|| format!("无法打开文件:"))?;

        let mmap = unsafe { Mmap::map(&file)? };
        println!(" 文件大小：{:.2}MB", mmap.len() as u64 / 1024 / 1024);
        Ok(MockData {
            width: 1023,
            height: 100,
            data: vec![0, 1, 2],
        })
    }
}
