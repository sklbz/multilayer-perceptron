use crate::linear_algebra::matrix::*;

pub(crate) trait GridDisplay {
    fn grid(&self) -> String;
    fn display(&self);
}

impl GridDisplay for Vector<f64> {
    fn grid(&self) -> String {
        let mut result = String::new();

        for i in self {
            result += &format!("{}\n", i);
        }
        result
    }
    fn display(&self) {
        println!("{}", self.grid());
    }
}

impl GridDisplay for Matrix<f64> {
    fn grid(&self) -> String {
        let mut result = String::new();

        for row in self {
            for i in row {
                result += &format!("{} ", i);
            }
            result += "\n";
        }
        result
    }

    fn display(&self) {
        println!("{}", self.grid());
    }
}

impl GridDisplay for Tensor<f64> {
    fn grid(&self) -> String {
        let mut result = String::new();

        for i in self {
            result += i.grid().as_str();
            result += "\n\n\n";
        }
        result
    }

    fn display(&self) {
        println!("{}", self.grid());
    }
}
