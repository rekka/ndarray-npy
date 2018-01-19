//! A simple test of the crate that creates a npy file in a temporary directory and tries to load
//! it in Python using `numpy.load`.
extern crate ndarray;
extern crate ndarray_npy;
extern crate tempdir;

use ndarray::prelude::*;
use std::fs::File;
use tempdir::TempDir;
use std::process::Command;

fn main() {
    let tmp_dir = TempDir::new("ndarray-npy").expect("create temp dir");
    let file_path = tmp_dir.path().join("test.npy");
    let mut tmp_file = File::create(file_path).expect("create temp file");

    let arr: Array2<f64> = Array2::from_shape_fn((3, 4), |(i, j)| (i + j) as f64);
    ndarray_npy::write(&mut tmp_file, &arr).unwrap();

    let output = Command::new("python")
        .current_dir(tmp_dir.path())
        .arg("-c")
        .arg("import numpy; print(numpy.load('test.npy'))")
        .output()
        .expect("run python");

    if output.status.success() {
        println!("Expected output:\n{}", arr);
        println!(
            "Python output:\n{}",
            String::from_utf8_lossy(&output.stdout)
        );
    } else {
        println!("{:?}", output);
    }
}
