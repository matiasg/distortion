extern crate ndarray;
extern crate num_traits;

pub mod distortion {
    use num_traits::{Float, ToPrimitive};
    use ndarray::Array2;

    pub struct FieldDistortion<'a, T>
        where T: Float,
              T: ToPrimitive,
    {
        dist_x: &'a Array2<T>,
        dist_y: &'a Array2<T>,
    }

    impl<'a, T: Float> FieldDistortion<'a, T> {
        pub fn new(distx: &'a Array2<T>, disty: &'a Array2<T>) -> FieldDistortion<'a, T> {
            return FieldDistortion{dist_x: distx, dist_y: disty};
        }

        pub fn get(&self,
            mat: Array2<i32>,
            xy: (usize, usize)) -> Option<f64> {
            let x = T::from(xy.0).unwrap() + *self.dist_x.get(xy).expect("outside bounds");
            let y = T::from(xy.1).unwrap() + *self.dist_y.get(xy).expect("outside bounds");

            let x0 = x.floor().to_usize()?;
            let xweight = (x - x.floor()).to_f64()?;

            let y0 = y.floor().to_usize()?;
            let yweight = (y - y.floor()).to_f64()?;

            let result = (1.0 - xweight) * (1.0 - yweight) * mat.get((x0, y0))?.to_f64()?
                + (1.0 - xweight) * yweight * mat.get((x0, y0 + 1))?.to_f64()?
                + xweight * (1.0 - yweight) * mat.get((x0 + 1, y0))?.to_f64()?
                + xweight * yweight * mat.get((x0 + 1, y0 + 1))?.to_f64()?
                ;

            Some(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr2,Array2};
    use crate::distortion::FieldDistortion;

    #[test]
    fn test_get() {
        let x = arr2(&[
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ]);

        let img = arr2(&[
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]);

        let dist = FieldDistortion::new(&x, &x);
        let g = dist.get(img, (1, 1)).unwrap();
        assert_eq!(g, 7.0);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds() {
        let x: Array2<f32> = arr2(&[[1.0]]);
        let img = arr2(&[[2]]);
        let dist = FieldDistortion::new(&x, &x);
        let _cant_get = dist.get(img, (1, 1));
    }
}
