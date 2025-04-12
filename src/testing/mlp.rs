use crate::multilayer_perceptron::multilayer_perceptron::MultiLayerPerceptron;

pub fn test() {
    let mlp = MultiLayerPerceptron::new(vec![2, 3, 1]);
    mlp.display();
}
