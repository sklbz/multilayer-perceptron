#[cfg(test)]
mod tests {
    use crate::mlp::multilayer_perceptron::{MultiLayerPerceptron, NeuralNetwork};
    use crate::mlp::utils::Database;

    // -----------------------------------------------------------------------
    // Dataset
    // -----------------------------------------------------------------------

    /// Génère `n` points (x, tanh(x)) sur [min, max].
    /// L'entrée est normalisée dans [-1, 1] via division par `max`.
    fn tanh_dataset(n: usize, min: f64, max: f64) -> Database {
        (0..n)
            .map(|i| {
                let x = min + (max - min) * (i as f64) / (n as f64 - 1.0);
                let input = vec![x / max];
                let target = vec![x.tanh()];
                let coefficient = 1.0 / n as f64;
                (input, target, coefficient)
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Utilitaire d'affichage
    // -----------------------------------------------------------------------

    fn print_table(net: &MultiLayerPerceptron, dataset: &Database, scale: f64) {
        println!(
            "\n{:<10} {:<12} {:<12} {:<10}",
            "x", "tanh(x)", "prédit", "erreur abs."
        );
        println!("{}", "-".repeat(46));
        for (input, target, _) in dataset {
            let predicted = net.calc(input.clone());
            let x_real = input[0] * scale;
            let err = (predicted[0] - target[0]).abs();
            println!(
                "{:<10.4} {:<12.6} {:<12.6} {:<10.6}",
                x_real, target[0], predicted[0], err
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 1 : l'erreur doit baisser significativement
    // -----------------------------------------------------------------------

    #[test]
    fn test_erreur_diminue() {
        let mut net = MultiLayerPerceptron::new(vec![1, 8, 8, 1]);
        let dataset = tanh_dataset(40, -3.0, 3.0);

        let avant = net.error_function(&dataset);
        net.backpropagation(&dataset, 5_000, 0.05);
        let apres = net.error_function(&dataset);

        println!("\nErreur avant : {:.6}", avant);
        println!("Erreur après : {:.6}", apres);

        assert!(
            apres < avant / 2.0,
            "L'erreur aurait dû au moins être divisée par 2 ({:.6} → {:.6})",
            avant,
            apres
        );
    }

    // -----------------------------------------------------------------------
    // Test 2 : tanh(0) = 0
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanh_zero() {
        let mut net = MultiLayerPerceptron::new(vec![1, 8, 8, 1]);
        let dataset = tanh_dataset(40, -3.0, 3.0);

        net.backpropagation(&dataset, 5_000, 0.05);

        let predicted = net.calc(vec![0.0]);
        println!("\ntanh(0) prédit : {:.6}  (attendu ≈ 0.0)", predicted[0]);

        assert!(
            predicted[0].abs() < 0.15,
            "tanh(0) devrait être proche de 0, obtenu : {:.6}",
            predicted[0]
        );
    }

    // -----------------------------------------------------------------------
    // Test 3 : tanh est antisymétrique — tanh(-x) ≈ -tanh(x)
    // -----------------------------------------------------------------------

    #[test]
    fn test_antisymetrie() {
        let mut net = MultiLayerPerceptron::new(vec![1, 8, 8, 1]);
        let dataset = tanh_dataset(40, -3.0, 3.0);

        net.backpropagation(&dataset, 5_000, 0.05);

        let scale = 3.0_f64;
        for x in [0.5_f64, 1.0, 1.5, 2.0] {
            let pos = net.calc(vec![x / scale])[0];
            let neg = net.calc(vec![-x / scale])[0];
            let somme = (pos + neg).abs();

            println!(
                "tanh({:.1}) ≈ {:.6}, tanh(-{:.1}) ≈ {:.6}, somme = {:.6}",
                x, pos, x, neg, somme
            );

            assert!(
                somme < 0.3,
                "Antisymétrie violée pour x={}: tanh(x) + tanh(-x) = {:.6}",
                x,
                somme
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 4 : affichage complet — informatif, passe toujours
    // Lancez : cargo test test_affichage -- --nocapture
    // -----------------------------------------------------------------------

    #[test]
    fn test_affichage_resultats() {
        let scale = 3.0_f64;
        let mut net = MultiLayerPerceptron::new(vec![1, 8, 8, 1]);
        let dataset = tanh_dataset(40, -scale, scale);

        println!("\n=== Courbe d'apprentissage ===");
        for step in 1..=5 {
            net.backpropagation(&dataset, 1_000, 0.05);
            let err = net.error_function(&dataset);
            println!("Itération {:>5} — erreur : {:.6}", step * 1_000, err);
        }

        println!("\n=== Résultats sur le dataset ===");
        print_table(&net, &dataset, scale);

        println!("\n=== Généralisation (points hors dataset) ===");
        println!(
            "{:<10} {:<12} {:<12} {:<10}",
            "x", "tanh(x)", "prédit", "erreur abs."
        );
        println!("{}", "-".repeat(46));
        for x in [-2.5_f64, -1.7, -0.6, 0.3, 1.2, 2.1] {
            let predicted = net.calc(vec![x / scale])[0];
            let expected = x.tanh();
            println!(
                "{:<10.4} {:<12.6} {:<12.6} {:<10.6}",
                x,
                expected,
                predicted,
                (predicted - expected).abs()
            );
        }
    }
}
