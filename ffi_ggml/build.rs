extern crate bindgen;

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=ggml.h");

    let bindings = bindgen::Builder::default()
        .header("ggml.h")
        .clang_arg("-std=c11")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .generate()
        .expect("Unable to generate bindings");

    bindings
        .write_to_file(PathBuf::from("./src/ggml_bindings.rs"))
        .expect("Couldn't write bindings!");
}
