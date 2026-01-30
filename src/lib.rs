mod mock;
pub fn run() {
    println!("1. mock 图像数据=============");
    let image_path = "618_door_001234.jpg";
    let image_rgb = mock::MockData::from_jpg(image_path);
}
