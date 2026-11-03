// 未見之題：以 Result、map 與 and_then 串接驗算。
fn positive(value: i32) -> Result<i32, String> {
    if value < 0 { Err(String::from("負數不合")) } else { Ok(value) }
}

fn lift(turns: i32, buckets: i32) -> Result<i32, String> {
    positive(turns).and_then(|a| positive(buckets).map(|b| a * b))
}

fn main() {
    assert_eq!(lift(3, 4), Ok(12));
    assert!(lift(-1, 4).is_err());
    assert_eq!(lift(2, 3).unwrap_or(0), 6);
}
