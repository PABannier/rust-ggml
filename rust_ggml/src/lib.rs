extern crate ffi_ggml;


#[derive(Copy, Debug)]
pub struct Context {
    ctx: ffi_ggml::ggml_context
}

impl Context {
    pub fn new(mem_size: u64, mem_buffer: &mut TODO, no_alloc: bool) -> Self {
        let params = ffi_ggml::ggml_init_params { mem_size, mem_buffer, no_alloc };
        Context { ctx: unsafe { ffi_ggml::ggml_init(params) } }
    }

    pub fn free(&mut self) {
        unsafe { ffi_ggml::ggml_free(&mut self.ctx) }
    }

    pub fn used_mem(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_used_mem(&self.ctx) };
        res
    }
}


#[derive(Copy, Debug)]
pub struct CGraph {
    graph: ffi_ggml::ggml_cgraph,
}

impl CGraph {
    pub fn new(n_threads: usize) -> Self {
        CGraph {
            graph: unsafe {
                ffi_ggml::ggml_cgraph {
                    n_nodes: 0,
                    n_leafs: 0,
                    n_threads,
                    work_size: 0,
                    work: None,
                    nodes: [],
                    grads: [],
                    leafs: [],
                    perf_runs: 0,
                    perf_cycles: 0,
                    perf_time_us: 0
                }
            },
        }
    }

    pub fn build_forward(tensor: &mut Tensor) -> Self {
        Self {
            graph: unsafe { ffi_ggml::ggml_build_forward(tensor) }
        }
    }

    pub fn build_backward(ctx: &mut Context, graph: &mut CGraph, keep: bool) -> Self {
        Self {
            graph: unsafe { ffi_ggml::ggml_build_backward(ctx, graph.graph, keep) }
        }
    }

    pub fn build_forward_expand(&mut self, tensor: &mut Tensor) {
        unsafe { ffi_ggml::ggml_build_forward_expand(&mut self.graph, tensor) } 
    }

    pub fn graph_compute(&mut self, ctx: &mut Context) {
        unsafe { ffi_ggml::ggml_graph_compute(ctx, &mut self.graph) }
    }

    pub fn graph_reset(&mut self) {
        unsafe { ffi_ggml::ggml_graph_reset(&mut self.graph) }
    }

    pub fn graph_print(&self) {
        unsafe { ffi_ggml::ggml_print(&self.graph) }
    }
}


#[derive(Copy, Debug)]
pub struct Tensor {
    ctx: &mut Context,
    tensor: ffi_ggml::ggml_tensor,
    dtype: DType,
    shape: &[usize],
}

impl Tensor {
    pub fn new(ctx: &mut Context, dtype: DType, n_dims: u8, shape: &[usize]) -> Result<Self, ()> {
        if n_dims > 4 || shape.len() > 4 {
            // Currently ggml only supports tensor up to 4 dimensions
            Err(())
        } else {
            Ok(Tensor {
                ctx,
                tensor: unsafe { ffi_ggml::ggml_new_tensor(ctx, tensor_type, n_dims, shape) },
                dtype,
                shape,
            })
        }
    }

    pub fn new_tensor_1d(ctx: &mut Context, dtype: DType, shape: usize) -> Self {
        Ok(Tensor {
            ctx,
            tensor: unsafe { ffi_ggml::ggml_new_tensor_1d(ctx, tensor_type, shape) },
            dtype,
            shape: &[shape],
        })
    }

    pub fn new_tensor_2d(ctx: &mut Context, dtype: DType, shape: &[usize]) {
        if shape.len() > 2 {
            Err(())
        } else {
            Ok(Tensor {
                ctx,
                tensor: unsafe { ffi_ggml::ggml_new_tensor_2d(ctx, tensor_type, shape[0], shape[1]) },
                dtype,
                shape,
            })
        }
    }

    pub fn new_tensor_3d(ctx: &mut Context, dtype: DType, shape: &[usize]) {
        if shape.len() > 3 {
            Err(())
        } else {
            Ok(Tensor {
                ctx,
                tensor: unsafe { ffi_ggml::ggml_new_tensor_3d(ctx, tensor_type, shape[0], shape[1], shape[2]) },
                dtype,
                shape,
            })
        }
    }

    pub fn new_tensor_4d(ctx: &mut Context, tensor_type: GGMLType, nelements: &[usize]) {
        if shape.len() > 4 {
            Err(())
        } else {
            Ok(Tensor {
                ctx,
                tensor: unsafe { ffi_ggml::ggml_new_tensor_4d(ctx, tensor_type, shape[0], shape[1], shape[2], shape[3]) },
                dtype,
                shape
            })
        }
    }

    pub fn nbytes(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_nbytes(self.tensor) };
        res
    }

    pub fn nelements(&self) -> i64 {
        let res = unsafe { ffi_ggml::ggml_nelements(self.tensor) };
        res
    }

    pub fn element_size(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_element_size(self.tensor) };
        res
    }

    /*
    TODO
    ----
    struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value);
    struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value);

    struct ggml_tensor * ggml_dup_tensor (struct ggml_context * ctx, const struct ggml_tensor * src);
    struct ggml_tensor * ggml_view_tensor(struct ggml_context * ctx, const struct ggml_tensor * src);

    struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
    struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value);
    struct ggml_tensor * ggml_set_f32 (struct ggml_tensor * tensor, float value);

    int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i);
    void    ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value);

    float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i);
    void  ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value);

    void * ggml_get_data    (const struct ggml_tensor * tensor);
    float * ggml_get_data_f32(const struct ggml_tensor * tensor);
    */
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Tensor {
            ctx: self.ctx,
            tensor: unsafe { ffi_ggml::ggml_dup_tensor(self.ctx, self.tensor) },
            dtype: self.dtype,
            shape: self.shape,
        }
    }
}

/// functions
/// helpers
/// ggml_cpy
///
/// Ops
/// ggml_norm
/// ggml_add
/// ggml_mul
/// ggml_repeat
/// ggml_rope
/// ggml_reshape_3d
/// ggml_mul_mat
/// ggml_transpose
/// ggml_view_1d
/// ggml_view_2d
/// ggml_permute
/// ggml_scale
/// ggml_soft_max
///
