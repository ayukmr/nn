use nalgebra::DVector;

pub type ActivationFn = fn(&DVector<f32>) -> DVector<f32>;

pub struct Activation {
    pub func:   ActivationFn,
    pub d_func: ActivationFn,
}

impl Activation {
    pub const NONE: Activation = Activation {
        func:   |x| x.clone(),
        d_func: |y| y.map(|_| 1.0),
    };

    pub const SIGMOID: Activation = Activation {
        func:   |x| x.map(|x| 1.0 / (1.0 + (-x).exp())),
        d_func: |y| y.map(|y| y * (1.0 - y)),
    };

    pub const TANH: Activation = Activation {
        func:   |x| x.map(|x| (x.exp() - (-x).exp()) / (x.exp() + (-x).exp())),
        d_func: |y| y.map(|y| 1.0 - y.powi(2)),
    };

    pub const RELU: Activation = Activation {
        func:   |x| x.map(|x| x.max(0.0)),
        d_func: |y| y.map(|y| if y > 0.0 { 1.0 } else { 0.0 }),
    };

    pub const SOFTMAX: Activation = Activation {
        func: |x| {
            let max  = x.max();
            let exps = x.map(|x| (x - max).exp());

            &exps / exps.sum()
        },
        d_func: |y| {
            y.map(|si| si * (1.0 - si))
        },
    };

    pub fn func(&self) -> ActivationFn {
        self.func
    }

    pub fn d_func(&self) -> ActivationFn {
        self.d_func
    }
}
