use tch::nn::{Module, OptimizerConfig};
use tch::{kind, nn, Device, Tensor};

fn my_module(p: nn::Path, dim: i64) -> impl nn::Module {
    let x1 = p.zeros("x1", &[dim]);
    let x2 = p.zeros("x2", &[dim]);
    nn::func(move |xs| xs * &x1 + xs.exp() * &x2)
}

fn gradient_descent() {
    let vs = nn::VarStore::new(Device::Cpu);
    let my_module = my_module(vs.root(), 7);
    let mut opt = nn::Sgd::default().build(&vs, 1e-2).unwrap();
    for _idx in 1..50 {
        // Dummy mini-batches made of zeros.
        let xs = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let ys = Tensor::zeros(&[7], kind::FLOAT_CPU);
        let loss = (my_module.forward(&xs) - ys).pow_tensor_scalar(2).sum(kind::Kind::Float);
        opt.backward_step(&loss);
    }
}

fn main() {
    let t = Tensor::of_slice(&[3, 1, 4, 1, 5]);
    let t = t * 2;
    t.print();
}
