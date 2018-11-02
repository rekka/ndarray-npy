# Deprecated! Use for example crate [ndarray-npy](https://crates.io/crates/ndarray-npy) for a much more complete implementation.

A simple serialization of ndarray arrays of simple types (`f32`, `f64`) into
[NumPy](http://www.numpy.org/)'s [`.npy`
format](https://docs.scipy.org/doc/numpy/neps/npy-format.html).

Files produced this way can be loaded with `numpy.load`.

```
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
```

For a more generic serialization and deserialization, see [crate
npy](https://crates.io/crates/npy).

