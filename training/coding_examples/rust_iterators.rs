// 以迭代器、Option 與 collect 揀合輪筒之數。
fn even_squares(limit: i32) -> Vec<i32> {
    (1..limit)
        .map(|value| value * value)
        .filter(|value| value % 2 == 0)
        .collect()
}

fn lookup(names: &[&str], target: &str) -> Option<usize> {
    names.iter().position(|name| *name == target)
}

fn main() {
    assert_eq!(even_squares(6), vec![4, 16]);
    let names = ["輪", "筒", "梘"];
    assert_eq!(lookup(&names, "筒"), Some(1));
    assert_eq!(lookup(&names, "帆").unwrap_or(99), 99);
}
