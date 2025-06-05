use crate::linear_algebra::matrix::*;

use super::multilayer_perceptron::MultiLayerPerceptron;

pub fn parse_params(input: String) -> (Vector<usize>, Tensor<f64>, Matrix<f64>) {
    let binding = input
        .replace("Architecture", "")
        .replace("Weights", "")
        .replace("Biases", "")
        .replace(",", "")
        .replace(":", "");
    let mut lines = binding.lines();

    let architecture: Vector<usize> = lines
        .next()
        .unwrap()
        .replace("[", "")
        .replace("]", "")
        .split_whitespace()
        .map(|s| s.parse::<usize>().unwrap())
        .collect();

    // This will not work
    let weights: Tensor<f64> = lines
        .next()
        .unwrap()
        .replace("]", "")
        .split("[")
        .map(|sup| -> Matrix<f64> {
            sup.split("[")
                .map(|sub| {
                    sub.split_whitespace()
                        .filter(|s| !s.is_empty())
                        .filter(|s| s != &" ")
                        .filter(|s| s != &"[")
                        .map(|s| match s.parse::<f64>() {
                            Ok(f) => f,
                            Err(_) => panic!("Error trying to parse: {}", s),
                        })
                        .collect()
                })
                .collect::<Matrix<f64>>()
        })
        .collect();

    let biases: Matrix<f64> = lines
        .next()
        .unwrap()
        .replace("]", "")
        .split("[")
        .map(|sub| sub.split("[").map(|s| s.parse::<f64>().unwrap()).collect())
        .collect();

    (architecture, weights, biases)
}

pub trait Save {
    fn save(&self, path: String);
    fn load(path: String) -> Self;
}

use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use bincode::{config, decode_from_std_read, encode_into_std_write};

impl Save for MultiLayerPerceptron {
    fn save(&self, path: String) {
        let config = config::standard();

        let path = Path::new(&path);
        let file = File::create(path).unwrap();
        let mut writer = BufWriter::new(file);

        encode_into_std_write(self, &mut writer, config).unwrap();
    }

    fn load(path: String) -> Self {
        let config = config::standard();

        let path = Path::new(&path);
        let file = File::open(path).unwrap();
        let mut reader = BufReader::new(file);

        decode_from_std_read(&mut reader, config).unwrap()
    }
}
