use anyhow::{Context, Result};
use jpeg_decoder::{Decoder};
use memmap2::Mmap;
/// mock data for image buffer
/// read from  jpg file
use std::{fs::File, io::Cursor, time::Instant};
pub struct MockData {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

impl MockData {
    pub fn from_jpg(path: &str) -> Result<Self> {
        let file = File::open(path).with_context(|| format!("无法打开文件:"))?;

        let mmap = unsafe { Mmap::map(&file)? };
        println!(" 文件大小：{:.2}MB", mmap.len() as u64 / 1024 / 1024);

        //判断是不是jpg
        let needle= vec![0xFF,0xD8,0xFF];
        if !mmap.starts_with(&needle) {
            anyhow::bail!("图片不是JPG格式！")
        }

        //  get image metadata
        let cursor = Cursor::new(&mmap);
        let mut decoder = Decoder::new(cursor);
        let _ = decoder.read_info();
        let (width, height, channels) =  match decoder.info() {
            Some(info) => (info.width as u32, info.height as u32, info.pixel_format.pixel_bytes() as  u8),
            None=>(0,0,3)
        };
        
        if width == 0 {
            anyhow::bail!("读取图片 元数据失败！")
        }

        // 开始解码
        println!("JPEG pixel format: 通道数{:?} 大小 {}*{}", channels, width, height);
        println!("----jpeg 开始解码----");
        let decodetimer = Instant::now();
        let  pixels = decoder.decode()?;
        println!("--------解码结束,耗时{}ms， 开始转为灰度数据",decodetimer.elapsed().as_millis());
        let graytimer = Instant::now();
        let mut gray_pixels = Vec::with_capacity((width as usize)*(height as  usize) );

        for chunk in pixels.chunks(channels as usize) {
            let gray = (0.229* (chunk[0]as f32) + 0.587*(chunk[1] as f32) + 0.114 * (chunk[2] as f32)).round() as  u8;
            gray_pixels.push(gray);
        }
        println!("====灰度数据转换完成，数据长度：{}, 内存大小{}MB,耗时：{}ms", gray_pixels.len(),(gray_pixels.len() as f32)/1024.0/1024.0,graytimer.elapsed().as_millis());
        Ok(MockData {
            width: width,
            height: height,
            data: gray_pixels,
        })
    }

    pub fn info(&self) {
        println!("info:: width:{},height:{},dataLen:{}",self.width,self.height,self.data.len());
    }
}
