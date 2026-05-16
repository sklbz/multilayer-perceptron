use crate::mlp::multilayer_perceptron::{MultiLayerPerceptron, NeuralNetwork};
use crate::mlp::utils::{Database, SharedVector, shared};
use std::time::Instant;

fn dummy_database(n: usize, input_size: usize, output_size: usize) -> Database {
    (0..n)
        .map(|i| {
            let x = i as f64 / n as f64;
            let input: SharedVector = shared(vec![x; input_size]);
            let target: SharedVector = shared(vec![x.tanh(); output_size]);
            let coefficient = 1.0 / n as f64;
            (input, target, coefficient)
        })
        .collect()
}

fn bench_architecture(layers: Vec<usize>, iterations: usize, db_size: usize) {
    let label = format!("{:?}", layers);
    let db = dummy_database(db_size, *layers.first().unwrap(), *layers.last().unwrap());
    let mut net = MultiLayerPerceptron::new(layers);

    let start = Instant::now();
    net.backpropagation(&db, iterations, 0.001);
    let elapsed = start.elapsed();

    println!(
        "Architecture {:<30} | {:>4} iter | {:>6} exemples | {:>8.3}s | {:>6.2}ms/iter",
        label,
        iterations,
        db_size,
        elapsed.as_secs_f64(),
        elapsed.as_millis() as f64 / iterations as f64,
    );
}

#[test]
fn test_bench_training_time() {
    let iterations = 10;
    let db_size = 32;

    println!("\n{}", "=".repeat(80));
    println!("Benchmark temps d'entraînement");
    println!("{}", "=".repeat(80));

    bench_architecture(vec![768, 1024, 1], iterations, db_size);
    bench_architecture(vec![768, 1536, 1], iterations, db_size);
    bench_architecture(vec![768, 1792, 1], iterations, db_size);
    bench_architecture(vec![1024, 1536, 1], iterations, db_size);
    bench_architecture(vec![1024, 1792, 1], iterations, db_size);
    bench_architecture(vec![1536, 1792, 1], iterations, db_size);
    bench_architecture(vec![768, 1024, 1539, 1792], iterations, 128);
}
