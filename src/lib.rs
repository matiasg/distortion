extern crate ndarray;
extern crate num_traits;
#[macro_use]
extern crate approx;

pub mod distortion {
    use ndarray::{s, Array, Array2, ArrayView, Axis};
    use num_traits::{Float, ToPrimitive};

    pub struct FieldDistortion<T>
    where
        T: Float,
        T: ToPrimitive,
    {
        dist_x: Array2<T>,
        dist_y: Array2<T>,
        accum: Array2<T>,
    }

    impl<'a, T: Float> FieldDistortion<T> {
        pub fn new(distx: Array2<T>, disty: Array2<T>) -> FieldDistortion<T> {
            FieldDistortion {
                accum: Array::<T, _>::ones(distx.raw_dim()),
                dist_x: distx,
                dist_y: disty,
            }
        }

        pub fn new_r(distx: &Array2<T>, disty: &Array2<T>) -> FieldDistortion<T> {
            let shape = (distx.shape()[0], distx.shape()[1]);
            FieldDistortion {
                dist_x: distx.clone(),
                dist_y: disty.clone(),
                accum: Array::<T, _>::ones(shape),
            }
        }

        pub fn get(&self, xy: (usize, usize)) -> Option<(T, T)> {
            let x = T::from(xy.0)? + *self.dist_x.get(xy).expect("outside bounds");
            let y = T::from(xy.1)? + *self.dist_y.get(xy).expect("outside bounds");
            Some((x, y))
        }

        pub fn apply(&self, mat: Array2<i32>, xy: (usize, usize)) -> Option<f64> {
            let (xdist, ydist) = self.get(xy)?;

            let x0 = xdist.floor().to_usize()?;
            let xweight = (xdist - xdist.floor()).to_f64()?;

            let y0 = ydist.floor().to_usize()?;
            let yweight = (ydist - ydist.floor()).to_f64()?;

            let result = (1.0 - xweight) * (1.0 - yweight) * mat.get((x0, y0))?.to_f64()?
                + (1.0 - xweight) * yweight * mat.get((x0, y0 + 1))?.to_f64()?
                + xweight * (1.0 - yweight) * mat.get((x0 + 1, y0))?.to_f64()?
                + xweight * yweight * mat.get((x0 + 1, y0 + 1))?.to_f64()?;

            Some(result)
        }
    }

    impl FieldDistortion<f32> {
        pub fn update(
            &mut self,
            original: (f32, f32),
            expected: (f32, f32),
            weight: f32,
        ) -> Option<()> {
            let WINDOW: usize = 4;

            let x0 = original.0.round().to_usize()?;
            let y0 = original.1.round().to_usize()?;

            let normal_window = Array::from_shape_fn((2 * WINDOW + 1, 2 * WINDOW + 1), |(i, j)| {
                f32::exp(
                    -((i as i32 - WINDOW as i32).pow(2) + (j as i32 - WINDOW as i32).pow(2)) as f32,
                )
            });

            let deltax = expected.0 - original.0;
            let deltay = expected.1 - original.1;

            let subslice = s![x0 - WINDOW..x0 + WINDOW + 1, y0 - WINDOW..y0 + WINDOW + 1];

            let mut slice = self.accum.slice_mut(subslice);
            slice += &normal_window.mapv(|v| v * weight);

            let mut slice = self.dist_x.slice_mut(subslice);
            slice += &normal_window.mapv(|v| v * weight * deltax);

            let mut slice = self.dist_y.slice_mut(subslice);
            slice += &normal_window.mapv(|v| v * weight * deltay);

            //let mut normal_window = Array2::<f32>::zeros((2 * WINDOW + 1, 2 * WINDOW + 1));
            //let mut i = -(WINDOW as i32);
            //for mut row in normal_window.lanes_mut(Axis(0)) {
            //    let mut j = -(WINDOW as i32);
            //    for e in row.iter_mut() {
            //        *e += f32::exp(-(i.pow(2) + j.pow(2)) as f32) ;
            //        j += 1;
            //    }
            //    i += 1;
            //}
            //let left = i32::max(0i32, x0 - WINDOW);
            //let right = i32::min(self.dist_x.shape()[1].to_i32()?, x0 + WINDOW);
            //let top = i32::max(0i32, y0 - WINDOW);
            //let bottom = i32::min(self.dist_x.shape()[0].to_i32()?, y0 + WINDOW);

            Some(())
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::distortion::FieldDistortion;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_get() {
        let x = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]);

        let img = arr2(&[[1, 2, 3], [4, 5, 6], [7, 8, 9]]);

        let dist = FieldDistortion::new_r(&x, &x);
        let g = dist.apply(img, (1, 1)).unwrap();
        assert_eq!(g, 7.0);
    }

    #[test]
    #[should_panic]
    fn test_out_of_bounds() {
        let x: Array2<f32> = arr2(&[[1.0]]);
        let img = arr2(&[[2]]);
        let dist = FieldDistortion::new_r(&x, &x);
        let _cant_get = dist.apply(img, (1, 1));
    }

    #[test]
    fn test_update() {
        let dx = Array2::<f32>::zeros((20, 20));
        let dy = Array2::<f32>::zeros((20, 20));
        let mut dist = FieldDistortion::new(dx, dy);

        dist.update((10., 12.), (11.1, 12.2), 1.0);

        assert_eq!(dist.get((10, 12)), Some((11.1, 12.2)));

        let xy = dist.get((11, 13)).expect("did not get value");
        assert_abs_diff_eq!(xy.0, 11.148, epsilon = 1e-3f32);
        assert_abs_diff_eq!(xy.1, 13.027, epsilon = 1e-3f32);
    }
}
