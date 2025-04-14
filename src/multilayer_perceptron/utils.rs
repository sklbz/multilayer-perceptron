pub fn into_layer(architecture: Vec<usize>) -> (Vec<usize>, Vec<usize>) {
    let layers_count = architecture.len() - 1;

    let rows: Vec<usize> = architecture
        .get(1..)
        .unwrap_or(&[])
        .to_vec()
        .into_iter()
        .map(|layer_size| layer_size as usize)
        .collect();

    let columns: Vec<usize> = architecture
        .get(..layers_count)
        .unwrap_or(&[])
        .to_vec()
        .into_iter()
        .map(|layer_size| layer_size as usize)
        .collect();

    (rows, columns)
}

pub fn square_error(result: &Vec<f64>, target: &Vec<f64>) -> f64 {
    result
        .iter()
        .zip(target.iter())
        .map(|(calc, target)| (calc - target).powf(2.0))
        .sum::<f64>()
}
