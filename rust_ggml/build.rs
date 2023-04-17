extern crate cc;

fn main() {
    println!("cargo:rerun-if-changed=ffi_ggml/ggml.c");
    cc::Build::new()
        .file("ffi_ggml/ggml.c")
        .flag("-Wno-unused-variable")
        .flag("-Wno-unused-function")
        .flag("-O3")
        .compile("ggml");
}
