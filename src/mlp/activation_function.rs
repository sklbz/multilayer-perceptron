use crate::linear_algebra::matrix::Vector;

pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
}

impl Activation {
    pub fn apply(&self, input: Vector<f64>) -> Vector<f64> {
        input
            .iter()
            .map(|coord: &f64| (self.function)(*coord))
            .collect::<Vector<f64>>()
    }

    pub fn gradient(&self, input: Vector<f64>) -> Vector<f64> {
        input
            .iter()
            .map(|partial: &f64| (self.derivative)(*partial))
            .collect::<Vector<f64>>()
    }
}
