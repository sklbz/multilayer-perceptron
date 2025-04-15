use crate::linear_algebra::generator::Generator;

#[allow(unused)]
pub(crate) fn test() {
    let vector = 3.generate_random();
    let matrix = (3, 2).generate_random();
    let tensor = (2, 3, 3).generate_random();
    println!("{:#?}", vector);
    println!("{:#?}", matrix);
    println!("{:#?}", tensor);

    let pseudo_vector = 3.generate_random();
    let pseudo_matrix = vec![3, 2, 1, 5].generate_random();
    let pseudo_tensor = vec![vec![2, 3, 3], vec![2, 3, 3]].generate_random();
    println!("{:#?}", pseudo_vector);
    println!("{:#?}", pseudo_matrix);
    println!("{:#?}", pseudo_tensor);
}
