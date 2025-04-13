use crate::multilayer_perceptron::multilayer_perceptron::MultiLayerPerceptron;

pub fn test() {
    let mlp = MultiLayerPerceptron::new(vec![2, 3, 1]);
    let calc = mlp.calc(vec![0.0, 0.0]);

    println!("\nCalc: {:?}", calc);
}
