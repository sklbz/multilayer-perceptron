use crate::mlp::multilayer_perceptron::{MultiLayerPerceptron, NeuralNetwork};

mod linear_algebra;
mod mlp;
mod test;

fn main() {
    let mlp = MultiLayerPerceptron::new(vec![1, 8, 8, 1]);
    mlp.save();

    let samples = 200;

    for i in 0..samples {
        let x = (500 * i) as f64 / (samples - 1) as f64;

        let y = mlp.calc(&[x])[0];

        println!("{},{}", x, y);
    }
}
