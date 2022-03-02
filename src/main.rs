extern crate ndarray;

use ndarray::{Array, Array2};
use distortion::distortion::FieldDistortion;

fn main() {
    let dx = Array2::<f32>::zeros((20, 20));
    let dy = Array2::<f32>::zeros((20, 20));
    let mut dist = FieldDistortion::new(dx, dy);

    dist.update((10., 12.), (11.1, 12.2), 1.0);

    let submat = Array::from_shape_fn((3, 3), |(i, j)| {
        dist.get((10 - 1 + i, 12 - 1 + j)).expect("oh, no")
    });

    println!("around (10, 12):\n{:?}", submat);

    let xy = dist.get((11, 13)).expect("did not get value");
    println!("around (10, 12): {:#?}", xy);
}
