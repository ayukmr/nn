mod utils;
use utils::load_data;

use nn::network::Network;
use nn::layer::Layer;
use nn::activation::Activation;

use std::fs::File;
use std::env;

use anyhow::Result;
use nalgebra::DVector;

pub fn main() -> Result<()> {
    let command = env::args().nth(1);

    match command.as_deref() {
        Some("train") => train_mnist()?,
        Some("test")  => test_mnist()?,

        _ => println!("usage: `mnist train`, `mnist test`"),
    }

    Ok(())
}

fn train_mnist() -> Result<()> {
    let train: Vec<(DVector<f32>, DVector<f32>)> =
        load_data(File::open("train.csv")?)?
            .into_iter()
            .map(|(label, input)| {
                let mut encoded = vec![0.0; 10];
                encoded[label] = 1.0;

                (DVector::from(encoded), input)
            })
            .collect();

    let mut network = Network::new(vec![
        Layer::new(784, 128, Activation::SIGMOID),
        Layer::new(128, 64,  Activation::SIGMOID),
        Layer::new(64,  10,  Activation::SOFTMAX),
    ]);

    println!("{}", network);

    let n_train = train.len() as f32;

    for epoch in 0..=10000 {
        let mut losses = 0.0;

        for (label, input) in &train {
            network.backprop(input, label);
            losses += network.loss(input, label);
        }

        println!("———");

        println!("epoch {}", epoch);
        println!("avg loss {}", losses / n_train);
    }

    network.save("model.json")?;

    Ok(())
}

pub fn test_mnist() -> Result<()> {
    let test = load_data(File::open("test.csv")?)?;

    let mut network = Network::new(vec![
        Layer::new(784, 128, Activation::SIGMOID),
        Layer::new(128, 64,  Activation::SIGMOID),
        Layer::new(64,  10,  Activation::SOFTMAX),
    ]);

    let mut correct   = 0;
    let mut incorrect = 0;

    network.load("model.json")?;

    for (label, input) in &test {
        let out = network.forward(&input);

        let num =
            out.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(num, _)| num)
                .unwrap();

        if num == *label {
            correct += 1;
        } else {
            incorrect += 1;
        }

        for row in input.as_slice().chunks(28) {
            for &val in row {
                let cell =
                    if      val > 0.8 { "█" }
                    else if val > 0.6 { "▓" }
                    else if val > 0.4 { "▒" }
                    else if val > 0.2 { "░" }
                    else { " " };

                print!("{}{}", cell, cell);
            }

            println!();
        }

        println!("prediction {}", num);
        println!("actual {}", label);

        println!("———");
    }

    println!();

    println!(
        "correct {} | incorrect {}",
        correct, incorrect,
    );

    println!(
        "accuracy {}",
        correct as f32 / (correct + incorrect) as f32,
    );

    Ok(())
}
