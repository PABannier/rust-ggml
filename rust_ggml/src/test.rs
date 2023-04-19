use crate::{CGraph, Context, GGML_DTYPE};

const MEM_SIZE: usize = 16 * 1024 * 1024;

macro_rules! binary_op_test {
    ($operation:ident, $input_a:expr, $input_b:expr, $ground_truth:expr) => {
        #[test]
        fn $operation() {
            let mut ctx = Context::new(MEM_SIZE, None, false).unwrap();
            let inp_a = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
            let inp_b = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
            let mut out = ctx.$operation(&inp_a, &inp_b);

            let mut graph = CGraph::build_forward(&mut out);

            inp_a.set_data_f32($input_a);
            inp_b.set_data_f32($input_b);

            graph.compute(&mut ctx);

            let res = out.get_data_f32_1d(0).unwrap();
            assert_eq!(res, $ground_truth);
        }
    };
}

macro_rules! unary_op_test {
    ($operation:ident, $input_a:expr, $ground_truth:expr) => {
        #[test]
        fn $operation() {
            let mut ctx = Context::new(MEM_SIZE, None, false).unwrap();
            let inp = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
            let mut out = ctx.$operation(&inp);

            let mut graph = CGraph::build_forward(&mut out);
            inp.set_data_f32($input_a);
            graph.compute(&mut ctx);

            let res = out.get_data_f32_1d(0).unwrap();
            assert_eq!(res, $ground_truth);
        }
    };
}

macro_rules! reduce_op_test {
    ($operation:ident, $input_a:expr, $ground_truth:expr) => {
        #[test]
        fn $operation() {
            let mut ctx = Context::new(MEM_SIZE, None, false).unwrap();
            let mut inp = ctx.new_tensor_1d(GGML_DTYPE::F32, 1).unwrap();
            let mut out = ctx.$operation(&inp);

            let mut graph = CGraph::build_forward(&mut out);
            inp.set_data_f32($input_a);
            graph.compute(&mut ctx);

            let res = out.get_data_f32_1d(0).unwrap();
            assert_eq!(res, $ground_truth);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    mod binary_op {
        use super::*;
        binary_op_test!(add, 2., 1., 3.);
        binary_op_test!(mul, 2., 1., 2.);
        binary_op_test!(sub, 3., 2., 1.);
        binary_op_test!(div, 4., 2., 2.);
        binary_op_test!(scale, 3., 2., 6.);
    }

    mod unary_op {
        use super::*;
        unary_op_test!(abs, -2.3, 2.3);
        unary_op_test!(sqr, -4., 16.);
        unary_op_test!(sqrt, 36., 6.);
        unary_op_test!(sgn, -3., -1.);
        unary_op_test!(neg, -3., 3.);
        // unary_op_test!(step, -3., 3.);
        unary_op_test!(relu, -3., 0.);
        unary_op_test!(gelu, 3.2, 3.1972656);
        unary_op_test!(silu, 4.7, 4.65625);
    }
}
