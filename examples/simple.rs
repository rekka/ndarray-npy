extern crate ndarray;
extern crate ndarray_npy;

use ndarray::prelude::*;
use std::io;

fn main() {
    let arr: Array2<f64> = Array2::zeros((3, 4));

    let stdout = io::stdout();
    let mut handle = stdout.lock();

    ndarray_npy::write(&mut handle, &arr).unwrap();
}
