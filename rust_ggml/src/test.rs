#[cfg(test)]
mod tests {
    use crate::{CGraph, Context, GGML_DTYPE};

    const MEM_SIZE: usize = 16 * 1024 * 1024;

    #[test]
    fn it_works() {
        let mut ctx = Context::new(MEM_SIZE, None, false).unwrap();
        let mut _a = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
        let mut _b = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
        let mut _c = ctx.add(&_a, &_b);

        let mut graph = CGraph::build_forward(&mut _c);

        _a.set_data_f32(2.);
        _b.set_data_f32(1.);

        graph.compute(&mut ctx);

        let res = _c.get_data_f32_1d(0).unwrap();
        assert_eq!(res, 3.);
    }

    #[test]
    fn add_works() {}

    #[test]
    fn mul_works() {}

    #[test]
    fn sub_works() {}

    #[test]
    fn div_works() {}

    #[test]
    fn repeat_works() {}

    #[test]
    fn rope_works() {}

    #[test]
    fn norm_works() {}

    #[test]
    fn reshape_3d_works() {}

    #[test]
    fn matmul_works() {}

    #[test]
    fn transpose_works() {}

    #[test]
    fn view_1d_works() {}

    #[test]
    fn view_2d_works() {}

    #[test]
    fn permute_works() {}

    #[test]
    fn scale_works() {}

    #[test]
    fn softmax_works() {}
}
