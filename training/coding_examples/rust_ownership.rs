// 以借用切片與 Result 明分可算、不可算之界。
#[derive(Clone, Copy)]
struct Reading {
    turns: i32,
    buckets: i32,
}

fn total(readings: &[Reading]) -> Result<i32, String> {
    let mut sum = 0;
    for reading in readings {
        if reading.turns < 0 || reading.buckets < 0 {
            return Err(String::from("負數不合"));
        }
        sum += reading.turns * reading.buckets;
    }
    Ok(sum)
}

fn doubled(values: &mut [i32]) {
    for value in values.iter_mut() {
        *value *= 2;
    }
}

fn main() -> Result<(), String> {
    let readings = [
        Reading { turns: 3, buckets: 4 },
        Reading { turns: 2, buckets: 2 },
    ];
    assert_eq!(total(&readings)?, 16);
    let mut counts = [1, 2, 3];
    doubled(&mut counts);
    assert_eq!(counts, [2, 4, 6]);
    assert!(total(&[Reading { turns: -1, buckets: 2 }]).is_err());
    Ok(())
}
