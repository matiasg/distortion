extern crate ndarray;

use ndarray::{arr1, arr2};

fn main() {
    let a = arr2(&[ [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]]);
    let b = arr2(&[ [9, 8, 7],
                    [6, 5, 4],
                    [3, 2, 1]]);
    let v = arr1(&[1, 4, 5]);

    println!("a + b = {}", &a + &b);
    println!("a * b = {:?}", &a.dot(&b));
    println!("v = {:?}", v);
    println!("a * v = {:?}", a.dot(&v));
}
