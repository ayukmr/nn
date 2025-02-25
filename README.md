# NN

Simple deep learning library in Rust.

Create a network by defining layers:
```rust
let mut network = Network::new(vec![
    Layer::new(256, 128, Activation::SIGMOID),
    Layer::new(128, 64,  Activation::SIGMOID),
    Layer::new(64,  10,  Activation::SOFTMAX),
]);
```

Train the network using `backprop`:
```rust
for _ in 0..1000 {
    for (label, input) in &train {
        network.backprop(input, label);
    }
}
```

Test the network using `forward`:
```rust
let (_, input) = &test[0];
let out = network.forward(input);
```

Evaluate the network using `loss`:
```rust
let (label, input) = &test[0];
let loss = network.loss(input, label);
```
