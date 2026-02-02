use std::{time::Instant};
use anyhow::{Ok, Result};
mod mock;
pub mod labelme;
mod roi;
pub fn run()->Result<(),anyhow::Error> {
    println!("1. mock 图像数据=============");
    let image_path = "618_door_001234.jpg";
    let mut image_rgb = mock::MockData::from_jpg(image_path)?;
    image_rgb.info();

    //读取切图参数
    let start = Instant::now();
    let mut label_config = labelme::LabelConfig::new();
    label_config.load_ignore_json("configs/roi_ignore/618_ignore.json")?;
    label_config.load_roi_json("configs/roi_merged/618.json")?;
    let end = start.elapsed();
    println!("加载切图配置：：{},耗时::{}ms",&label_config,end.as_millis());

    // ignor 区域像素置为背景
    for rect in label_config.ignore_rects.iter() {
        let start_line = (rect.y).min(0);
        let end_line= (rect.y+rect.height).min(image_rgb.height as usize);
        let start_x = (rect.x).min(0);
        let end_x = (rect.x+rect.width).min(image_rgb.width as usize);

        for line in start_line ..= end_line {
            let start = line * (image_rgb.width as usize) + start_x;
            let end = line* (image_rgb.width as usize)  + end_x;
            let slice= &mut image_rgb.data[start ..end+1];
            slice.fill(0);
        }
    }

    let mut _session = roi::RoiSession::create_session()?;
    let mut crop_data = Vec::<f32>::with_capacity(1024*1024);
    let full_width = image_rgb.width  as usize;
    // let full_height = image_rgb.height as usize;
    let start1 = Instant::now();
    for rect in   label_config.roi_rects.iter() {
        // let crop_w:usize = 1024;
        let strider:usize = 896;
        let n_x = (0 .. rect.width ).step_by(strider).len();
        let n_y = (0 .. rect.height).step_by(strider).len();
        println!("共有{}个 1024切片",n_x*n_y);
        for y in  (0 .. rect.height ).step_by(strider) {
            for x in (0 .. rect.width).step_by(strider) {
                let start = Instant::now();

                for i in 0 .. 1024 {
                    if i+y >= rect.height {
                        crop_data[0..1024].fill(0.0);
                    }else {
                        let start_x =(y+i)*full_width + rect.x+x;
                        let end_x = (start_x + 1024).min((y+i)*full_width+rect.x+x +rect.width);
                        let rowdata = &image_rgb.data[start_x .. end_x];
                        crop_data.extend(rowdata.iter().map(|&s| (s as f32)/255.0));
                        if rowdata.len() < 1024 {
                            crop_data[rowdata.len() .. 1024-rowdata.len()].fill(0.0);
                        }
                    }
                }
                let end = start.elapsed();
                let byte_size = std::mem::size_of_val(crop_data.as_slice());
                let mb_size = byte_size as f64 / (1024.0 * 1024.0);
                println!("切片数据 ：：：： {}mb,耗时：：{}us",mb_size,end.as_micros());
                let _ =_session.run_with_vec(&crop_data);
                crop_data.clear();
            }
        }
        
    }
    let end1 = start1.elapsed();
    println!("推理完成：： 耗时 {}ms",end1.as_millis());
    

    Ok(())
}
 