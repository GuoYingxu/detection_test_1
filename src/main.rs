use detection_test_1::run;
use detection_test_1::labelme::LabelConfig;

fn main() {
    println!("Hello, world!");

    // 测试 LabelConfig 的 Display 实现
    let config = LabelConfig::new();
    println!("Empty LabelConfig: {}", config);

    // 继续执行原始功能
    if let Err(e) = run() {
        eprintln!("Error: {}", e);
    }
}
