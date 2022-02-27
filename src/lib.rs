extern crate ndarray;
extern crate num_traits;

pub mod distortion {
    use num_traits::Float;
    use ndarray::{arr2, Array2};

    pub struct FieldDistortion<'a, T>
        where T: Float,
    {
        dist_x: &'a Array2<T>,
        dist_y: &'a Array2<T>,
    }

    impl<'a, T: Float> FieldDistortion<'a, T> {
        pub fn new(distx: &'a Array2<T>, disty: &'a Array2<T>) -> FieldDistortion<'a, T> {
            return FieldDistortion{dist_x: distx, dist_y: disty};
        }

        pub fn get(&self, mat: Array2<i32>, xy: (usize, usize)) -> Option<(T, T)> {
            let x = T::from(xy.0)? + *self.dist_x.get(xy)?;
            let y = T::from(xy.1)? + *self.dist_y.get(xy)?;
            Some((x, y))
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr2;
    use crate::distortion;

    #[test]
    fn test_get() {
        let x = arr2(&[
            [1.1, 2.2, 3.3],
            [4.4, 5.5, 6.6],
            [7.7, 8.8, 9.9],
        ]);

        let img = arr2(&[
            [1, 0, 1],
            [0, 0, 1],
            [1, 2, 3],
        ]);

        let dist = distortion::FieldDistortion::new(&x, &x);
        let g = dist.get(img, (1, 1)).unwrap();
        assert_eq!(g, (6.5, 6.5));
    }
}
