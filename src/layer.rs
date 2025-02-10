use crate::activation::Activation;

use std::fmt;

use nalgebra::{DMatrix, DVector};
use rand::Rng;

pub struct Layer {
    in_size:  usize,
    out_size: usize,

    activation: Activation,

    pub weights: DMatrix<f32>,
    pub biases:  DVector<f32>,
}

impl Layer {
    pub fn new(
        in_size:    usize,
        out_size:   usize,
        activation: Activation,
    ) -> Self {
        let mut rng = rand::rng();

        let weights = DMatrix::from_fn(
            out_size, in_size,
            |_, _| rng.random_range(-1.0..1.0),
        );

        let biases = DVector::from_fn(
            out_size,
            |_, _| rng.random_range(-1.0..1.0),
        );

        Self {
            in_size,
            out_size,

            activation,

            weights,
            biases,
        }
    }

    pub fn sum(&self, input: &DVector<f32>) -> DVector<f32> {
        &self.weights * input + &self.biases
    }

    pub fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        self.activation(&self.sum(input))
    }

    pub fn activation(&self, x: &DVector<f32>) -> DVector<f32> {
        self.activation.func()(x)
    }

    pub fn d_activation(&self, x: &DVector<f32>) -> DVector<f32> {
        let y = self.activation(x);
        self.activation.d_func()(&y)
    }

    pub fn weights(&self) -> &DMatrix<f32> {
        &self.weights
    }

    pub fn biases(&self) -> &DVector<f32> {
        &self.biases
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Layer ({}, {})", self.in_size, self.out_size)
    }
}
