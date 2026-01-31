use std::time::Instant;

use anyhow::{Ok, Result};
use image::math::Rect;

mod mock;
pub mod labelme;
mod roi;
pub fn run()->Result<(),anyhow::Error> {
    println!("1. mock 图像数据=============");
    let image_path = "618_door_001234.jpg";
    let image_rgb = mock::MockData::from_jpg(image_path)?;
    image_rgb.info();

    //读取切图参数
    let start = Instant::now();
    let mut label_config = labelme::LabelConfig::new();
    label_config.load_ignore_json("configs/roi_ignore/618_ignore.json")?;
    label_config.load_roi_json("configs/roi_merged/618.json")?;
    let end = start.elapsed();
    println!("加载切图配置：：{},耗时::{}ms",&label_config,end.as_millis());

    let session = roi::RoiSession::create_session()?;
    
    for rect in   label_config.roi_rects.iter() {

    }
    

    Ok(())
}
 