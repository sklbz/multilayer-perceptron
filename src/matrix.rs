use rand::prelude::*;

pub type Matrix<T> = Vec<Vec<T>>;

trait MatrixTrait {
    fn random(rows: usize, cols: usize) -> Self;
}

impl MatrixTrait for Matrix<f64> {
    fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::rng();

        let mut matrix = Vec::new();
        for _ in 0..rows {
            let mut row = Vec::new();
            for _ in 0..cols {
                row.push(rng.random::<f64>());
            }
            matrix.push(row);
        }
        matrix
    }
}
