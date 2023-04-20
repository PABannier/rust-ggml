# ggml-rust

ggml-rust is a Rust crate that provides safe and fast bindings to the C tensor computing library called [ggml](https://github.com/ggerganov/ggml). With ggml-rust, you can leverage the power of ggml's tensor operations in your Rust applications, while enjoying Rust safety.

## Features

- Safe: ggml-rust provides a safe Rust API that helps prevent null pointer dereferences and buffer overflows, through Rust's strong type system and memory safety guarantees.
- Fast: ggml-rust leverages the high-performance tensor operations of ggml, allowing you to efficiently perform computations on multi-dimensional arrays (up to 4 dimensions) of data.

## Installation

ggml-Rust requires the ggml C library to be installed on your system. You can install ggml using the following steps:

1. Download the latest version of ggml from the official ggml GitHub repository: [https://github.com/ggml/ggml](https://github.com/ggml/ggml).
2. Follow the installation instructions provided by ggml to build and install the C library on your system.

Once you have ggml installed on your system, you can add ggml-Rust as a dependency in your Rust project's `Cargo.toml` file:

```toml
[dependencies]
ggml-rust = "0.1.0"
```

## Usage

To use ggml-Rust in your Rust project, simply import it and start using its APIs:

```rust
use ggml_rust::\*;

fn main() {
    const MEM_SIZE = 16 * 1024 * 1024;  // Allocate enough memory
    // Setup a context
    let mut ctx = Context::new(MEM_SIZE, None, false).unwrap();

    // Define the computational graphs
    let inp_a = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
    let inp_b = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
    let mut out = ctx.$operation(&inp_a, &inp_b);

    let mut graph = CGraph::build_forward(&mut out);

    // Input the graph with data
    inp_a.set_data_f32($input_a);
    inp_b.set_data_f32($input_b);

    // Forward pass
    graph.compute(&mut ctx);

    // Get result
    let res = out.get_data_f32_1d(0).unwrap();
}
```

Please refer to the API documentation for more detailed information on how to use ggml-Rust.

## Organization

The raw bindings are generated with `rust-bindgen` and stored in `rust_ggml/ffi_ggml` by running
`cargo build`. The `rust-ggml` crate is built to be a safe wrapper around the unsafe bindings generated
by `rust-bindgen`.

## Contributing

If you would like to contribute to ggml-Rust, please see our Contribution Guidelines for more information.

## License

ggml-Rust is released under the MIT License.
