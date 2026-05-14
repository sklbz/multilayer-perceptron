pub fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

pub fn RELU(v: Vec<f64>) -> Vec<f64> {
    v.iter()
        .map(|coord: &f64| relu(*coord))
        .collect::<Vec<f64>>()
}
