use crate::layer::Layer;

use std::{fmt, fs};

use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use serde::{Serialize, Deserialize};

pub struct Network {
    layers: Vec<Layer>,
}

#[derive(Serialize, Deserialize)]
pub struct State {
    pub weights: Vec<DMatrix<f32>>,
    pub biases:  Vec<DVector<f32>>,
}

impl Network {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let (weights, biases) =
            self.layers
                .iter()
                .map(|l| (l.weights.clone(), l.biases.clone()))
                .unzip();

        let state = State { weights, biases };
        let data  = serde_json::to_string(&state)?;

        fs::write(path, data)?;

        Ok(())
    }

    pub fn load(&mut self, path: &str) -> Result<()> {
        let data = &fs::read_to_string(path)?;
        let state: State = serde_json::from_str(data)?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.weights = state.weights[i].clone();
            layer.biases  = state.biases[i].clone();
        }

        Ok(())
    }

    pub fn forward(&self, input: &DVector<f32>) -> DVector<f32> {
        self.layers.iter().fold(
            input.clone(),
            |x, layer| layer.forward(&x)
        )
    }

    pub fn loss(&self, input: &DVector<f32>, desired: &DVector<f32>) -> f32 {
        (self.forward(input) - desired).abs().sum()
    }

    pub fn backprop(&mut self, input: &DVector<f32>, desired: &DVector<f32>) {
        // forward with input
        let (a, da_z): (Vec<DVector<f32>>, Vec<DVector<f32>>) =
            self.layers.iter().fold(
                (vec![input.clone()], vec![input.clone()]),
                |(mut a, mut da_z), layer| {
                    let x = a.last().unwrap();
                    let y = &layer.sum(x);

                    da_z.push(layer.d_activation(y));
                    a.push(layer.activation(y));

                    (a, da_z)
                },
            );

        let a_ref    = &a;
        let da_z_ref = &da_z;

        // ∂a_k^l/∂C for each neuron k in layer l
        let d_n: Vec<Vec<f32>> =
            self.layers
                .iter()
                .enumerate()
                .rev()
                .fold(
                    vec![
                        (2.0 * (
                            a.last().unwrap() - desired
                        )).as_slice().to_owned()
                    ],
                    |mut acc, (l, layer)| {
                        // l is actually at l
                        // layer is at l + 1

                        let acc_ref = &acc;

                        // ∂C/∂a_k for each neuron k
                        let d_l = (0..layer.weights().ncols()).map(move |k| {
                            // ∑∂C/∂w_jk for each weight k -> j
                            (0..layer.weights().nrows()).fold(
                                0.0,
                                move |acc, j| {
                                    // w_kj^(l + 1)
                                    let w = layer.weights()[(j, k)];

                                    // ∂C/∂a_j^(l + 1)
                                    let d = &acc_ref[0][j];

                                    // w_kj^(l + 1) * σ'(z_j^(l + 1)) * ∂C/∂a_j^(l + 1)
                                    let sum = w * da_z_ref[l + 1][j] * d;

                                    acc + sum
                                },
                            )
                        }).collect();

                        acc.insert(0, d_l);
                        acc
                    }
                )
                .into_iter()
                .collect();

        let d_n_ref = &d_n;

        // move -∂w_jk^l/∂C for each weight k -> j in layer l
        self.layers.iter_mut().enumerate().for_each(|(l, layer)| {
            // l is actually at l - 1
            // layer is at l

            for k in 0..layer.weights().ncols() {
                for j in 0..layer.weights().nrows() {
                    // a_k^(l - 1)
                    let a = a_ref[l][k];

                    // ∂C/∂a_j^l
                    let d = &d_n_ref[l + 1][j];

                    // a_k^(l - 1) * σ'(z_j^l) * ∂C/∂a_j^l
                    let d_w = a * da_z_ref[l + 1][j] * d;

                    // change weight k -> j by -∂C/∂w_jk^l
                    layer.weights[(j, k)] -= d_w * 1e-4;
                }
            }
        });

        // move -∂b_j^l/∂C for each bias j in layer l
        self.layers.iter_mut().enumerate().for_each(|(l, layer)| {
            // l is actually at l - 1
            // layer is at l

            for j in 0..layer.biases().ncols() {
                // ∂C/∂a_j^l
                let d = &d_n_ref[l + 1][j];

                // σ'(z_j^l) * ∂C/∂a_j^l
                let d_b = da_z_ref[l + 1][j] * d;

                // change bias j by -∂C/∂b_j^l
                layer.biases[j] -= d_b * 1e-4;
            }
        });
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Network {{")?;
        for layer in &self.layers {
            writeln!(f, " {}", layer)?;
        }
        write!(f, "}}")?;

        Ok(())
    }
}
