#[test]
fn simple_test() {
    use multilayer_perceptron::mlp::activation_function::Activation;

    let acti = Activation {
        function: |x: f64| x,
        derivative: |x: f64| 1,
    };

    assert_eq!(acti.apply(vec![-1.])[0], -1.);
}
