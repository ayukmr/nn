use nn::network::Network;
use nn::layer::Layer;
use nn::activation::Activation;

use anyhow::Result;
use nalgebra::dvector;

fn main() -> Result<()> {
    let input   = dvector![1.0, 2.0, 3.0, 4.0, 5.0];
    let desired = dvector![1.0, 0.5, 0.0];

    let mut network = Network::new(vec![
        Layer::new(5,  16, Activation::TANH),
        Layer::new(16, 16, Activation::TANH),
        Layer::new(16, 3,  Activation::TANH),
    ]);

    println!("{}", network);

    for epoch in 0..=50000 {
        network.backprop(&input, &desired);

        if epoch % 200 == 0 {
            let out  = network.forward(&input);
            let loss = network.loss(&input, &desired);

            println!("———");

            println!("epoch {}", epoch);
            println!("loss {}", loss);
            println!("{}", out);
        }
    }

    let out  = network.forward(&input);
    let loss = network.loss(&input, &desired);

    println!("———");

    println!("final");
    println!("loss {}", loss);
    println!("{}", out);

    Ok(())
}
