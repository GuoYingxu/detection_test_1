use std::{fmt, path::Path};
use anyhow::{Ok, Result, anyhow};
use serde::Deserialize;
use serde_json;

 
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect { 
    pub x:usize,
    pub y:usize,
    pub width:usize,
    pub height:usize,
}
impl Rect {
    pub fn new(x:usize,y:usize,width:usize,height:usize)->Self {
        Self{x,y,width,height}
    }
}
#[derive(Debug, Clone, Deserialize)]
pub struct LabelmeShape {
    pub points:Vec<Vec<f32>>,
    pub shape_type:String,
}
#[derive(Debug, Clone, Deserialize)]
pub struct IgnoreConfig { 
    shapes:Vec<LabelmeShape>,
}
pub struct LabelConfig {
    pub ignore_rects:Vec<Rect>, // 忽略区域
    pub roi_rects:Vec<Rect> //推理区域   
}
impl LabelConfig {
    pub fn new()->Self {
        Self { ignore_rects: Vec::new(),roi_rects:Vec::new()}
    }
    pub fn load_ignore_json<P: AsRef<Path>>(&mut self,path:P)->Result<&mut Self> {
        let json_content = std::fs::read_to_string(path)?;
        let ignore_config:IgnoreConfig = serde_json::from_str(&json_content)?;
        let rects = ignore_config.shapes.iter()
            .filter(|shape| shape.shape_type == "rectangle")
            .map(|shape| {
                // 检查shape 是否合规 （2个点）
                if shape.points.len()!=2 {
                    return Err(anyhow!(format!(
                        "矩形形状的 points 数量错误，需要 2 个点，实际 {} 个",
                        shape.points.len()
                        )));
                }

                //检查每个点是否包含 xy 两个坐标
                let p1 = &shape.points[0];
                let p2 = &shape.points[1];
                if p1.len() !=2 || p2.len() !=2 {
                    return Err(anyhow!("点坐标格式错误,需要【x,y】格式"));
                }

                // 计算宽高
                let width = (p2[0] - p1[0]).abs();
                let height = (p2[1] - p1[1]).abs();

                let x = p1[0].min(p2[0]).floor() as usize;
                let y = p1[1].min(p2[1]).ceil() as  usize;

                Ok(Rect::new(x,y,width.ceil() as usize,height.ceil() as  usize))
            }).collect::<Result<Vec<_>>>()?;
            self.ignore_rects = rects;
            Ok(self)
    }
    
    pub fn load_roi_json<P: AsRef<Path>>(&mut self,path:P)->Result<&mut Self> {
        let json_content = std::fs::read_to_string(path)?;
        let merge_config:IgnoreConfig = serde_json::from_str(&json_content)?;
        let rects = merge_config.shapes.iter()
            .filter(|shape| shape.shape_type == "rectangle")
            .map(|shape| {
                // 检查shape 是否合规 （2个点）
                if shape.points.len()!=2 {
                    return Err(anyhow!(format!(
                        "矩形形状的 points 数量错误，需要 2 个点，实际 {} 个",
                        shape.points.len()
                        )));
                }
                //检查每个点是否包含 xy 两个坐标
                let p1 = &shape.points[0];
                let p2 = &shape.points[1];
                if p1.len() !=2 || p2.len() !=2 {
                    return Err(anyhow!("点坐标格式错误,需要【x,y】格式"));
                }

                // 计算宽高
                let width = (p2[0] - p1[0]).abs() ;
                let height = (p2[1] - p1[1]).abs();

                let x = p1[0].min(p2[0]).floor() as  usize;
                let y = p1[1].min(p2[1]).ceil() as usize;

                Ok(Rect::new(x,y,width.ceil() as usize,height.ceil() as usize))
            }).collect::<Result<Vec<_>>>()?;
            self.roi_rects = rects;
            Ok(self)
    }
    
    
}

impl fmt::Display for LabelConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LabelConfig {{ ignore_rects: {:?}, roi_rects: {:?} }}",
               self.ignore_rects, self.roi_rects)
    }
}