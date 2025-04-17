autodiff - rust
===============

this crate implements a minimal reverse-mode autodiff engine in rust. it’s designed for low-level control, zero-cost abstractions, and aligned storage suitable for SIMD-heavy computation with minimal dependencies.

motivations:
- understand backprop from scratch
- get an ergonomic tensor abstraction that compiles down to raw ptrs
- experiment with graph-based autodiff and memory reuse

status:  
WIP. tensor creation, shape inference, and raw aligned storage implemented.

roadmap:
- [x] tensor initialization + shape parsing
- [x] avx2-friendly aligned heap storage
- [ ] shape rewrite
- [ ] matrix ops
- [ ] computation graph w/ petgraph (or maybe a custom DAG enabled with feature flags)
- [ ] operator overloading
- [ ] backward pass
- [ ] optimization passes (e.g. buffer reuse)


this is NOT:
- a pytorch clone
- ndarray with gradient tape
- made for GPUs!
- production-ready

this *is*:
- a fun way to use unsafe, low-level rust
- an experiment with raw SIMD
- a journey to learn why pytorch made the decisions it did

usage: 

```rust
use autodiff::Tensor;
let a = Tensor::new([1.0, 2.0, 3.0, 4.0])?;
let b: Tensor<f32> = vec![[vec![1.0, 2.0], vec![3.0, 4.0]]].to_tensor()?;
let c = a.dot(b) // not yet implemented
```