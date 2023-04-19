#![allow(non_camel_case_types)]

extern crate ffi_ggml;

use std::ffi::c_void;

#[repr(u32)]
pub enum GGML_DTYPE {
    F32 = 0,   // f32
    F16 = 1,   // f16
    Q4_0 = 2,  // quantized 4 bytes
    Q4_1 = 3,  // ????
    Q8_0 = 4,  // quantized 8 bytes
    I8 = 5,    // i8
    I16 = 6,   // i16
    I32 = 7,   // i32
    COUNT = 8, // ???
}

pub struct Context {
    ctx: *mut ffi_ggml::ggml_context,
}

macro_rules! safe_bindings_op {
    ($name:ident, $ffi_fn:path $(, $arg:ident)*) => {
        pub fn $name(&self, $($arg: &Tensor),*) -> Tensor {
            Tensor {
                tensor: unsafe { $ffi_fn(self.ctx, $($arg.tensor),*) },
            }
        }
    }
}

impl Context {
    pub fn new(mem_size: usize, mem_buffer: Option<&mut [u8]>, no_alloc: bool) -> Result<Self, ()> {
        let params = ffi_ggml::ggml_init_params {
            mem_size: mem_size.try_into().unwrap(),
            mem_buffer: match mem_buffer {
                Some(buffer) => buffer.as_mut_ptr() as *mut c_void,
                None => std::ptr::null_mut(),
            },
            no_alloc,
        };
        Ok(Context {
            ctx: unsafe { ffi_ggml::ggml_init(params) },
        })
    }

    pub fn used_mem(&self) -> u64 {
        let res = unsafe { ffi_ggml::ggml_used_mem(self.ctx) };
        res
    }

    pub fn new_tensor_1d(&self, dtype: GGML_DTYPE, shape: usize) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_new_tensor_1d(self.ctx, dtype as u32, shape.try_into().unwrap())
            },
        })
    }

    pub fn new_tensor_2d(&self, dtype: GGML_DTYPE, shape: &[usize]) -> Result<Tensor, ()> {
        if shape.len() > 2 {
            Err(())
        } else {
            Ok(Tensor {
                tensor: unsafe {
                    ffi_ggml::ggml_new_tensor_2d(
                        self.ctx,
                        dtype as u32,
                        shape[0].try_into().unwrap(),
                        shape[1].try_into().unwrap(),
                    )
                },
            })
        }
    }

    pub fn new_tensor_3d(&self, dtype: GGML_DTYPE, shape: &[usize]) -> Result<Tensor, ()> {
        if shape.len() > 3 {
            Err(())
        } else {
            Ok(Tensor {
                tensor: unsafe {
                    ffi_ggml::ggml_new_tensor_3d(
                        self.ctx,
                        dtype as u32,
                        shape[0].try_into().unwrap(),
                        shape[1].try_into().unwrap(),
                        shape[2].try_into().unwrap(),
                    )
                },
            })
        }
    }

    pub fn new_tensor_4d(&self, dtype: GGML_DTYPE, shape: &[usize]) -> Result<Tensor, ()> {
        if shape.len() > 4 {
            Err(())
        } else {
            Ok(Tensor {
                tensor: unsafe {
                    ffi_ggml::ggml_new_tensor_4d(
                        self.ctx,
                        dtype as u32,
                        shape[0].try_into().unwrap(),
                        shape[1].try_into().unwrap(),
                        shape[2].try_into().unwrap(),
                        shape[3].try_into().unwrap(),
                    )
                },
            })
        }
    }

    pub fn new_i32(&self, value: i32) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe { ffi_ggml::ggml_new_i32(self.ctx, value) },
        })
    }

    pub fn new_f32(&self, value: f32) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe { ffi_ggml::ggml_new_f32(self.ctx, value) },
        })
    }

    pub fn duplicate(&self, a: &Tensor) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe { ffi_ggml::ggml_dup_tensor(self.ctx, a.tensor) },
        })
    }

    safe_bindings_op!(abs, ffi_ggml::ggml_abs, a);

    // check done at GGML level
    safe_bindings_op!(add, ffi_ggml::ggml_add, a, b);

    safe_bindings_op!(cont, ffi_ggml::ggml_cont, a);

    // check done at GGML level
    safe_bindings_op!(conv_1d_1s, ffi_ggml::ggml_conv_1d_1s, a, b);

    // check done at GGML level
    safe_bindings_op!(conv_1d_2s, ffi_ggml::ggml_conv_1d_2s, a, b);

    // check done at GGML level
    safe_bindings_op!(sub, ffi_ggml::ggml_sub, a, b);

    // check done at GGML level
    safe_bindings_op!(mul, ffi_ggml::ggml_mul, a, b);

    // check done at GGML level
    safe_bindings_op!(div, ffi_ggml::ggml_div, a, b);

    safe_bindings_op!(sqr, ffi_ggml::ggml_sqr, a);

    safe_bindings_op!(sqrt, ffi_ggml::ggml_sqrt, a);

    safe_bindings_op!(sum, ffi_ggml::ggml_sum, a);

    safe_bindings_op!(mean, ffi_ggml::ggml_mean, a);

    // check done at GGML level
    safe_bindings_op!(repeat, ffi_ggml::ggml_repeat, a, b);

    safe_bindings_op!(sgn, ffi_ggml::ggml_sgn, a);

    safe_bindings_op!(neg, ffi_ggml::ggml_neg, a);

    safe_bindings_op!(step, ffi_ggml::ggml_step, a);

    safe_bindings_op!(relu, ffi_ggml::ggml_relu, a);

    safe_bindings_op!(gelu, ffi_ggml::ggml_gelu, a);

    safe_bindings_op!(silu, ffi_ggml::ggml_silu, a);

    safe_bindings_op!(norm, ffi_ggml::ggml_norm, a);

    safe_bindings_op!(rms_norm, ffi_ggml::ggml_rms_norm, a);

    // check done at GGML level
    safe_bindings_op!(mul_mat, ffi_ggml::ggml_mul_mat, a, b);

    safe_bindings_op!(scale, ffi_ggml::ggml_scale, a, b);

    // check done at GGML level
    safe_bindings_op!(reshape, ffi_ggml::ggml_reshape, a, b);

    safe_bindings_op!(transpose, ffi_ggml::ggml_transpose, a);

    safe_bindings_op!(get_rows, ffi_ggml::ggml_get_rows, a, b);

    safe_bindings_op!(softmax, ffi_ggml::ggml_soft_max, a);

    // pub fn map_unary(
    //     &self,
    //     a: &Tensor,
    //     unary_op: Option<unsafe fn(i32, &mut f32, &f32)>,
    // ) -> Result<Tensor, ()> {
    //     // TODO: check type f32
    //     Ok(Tensor {
    //         tensor: unsafe { ffi_ggml::ggml_map_unary_f32(self.ctx, a.tensor, unary_op) },
    //     })
    // }

    // pub fn map_binary(
    //     &self,
    //     a: &Tensor,
    //     b: &Tensor,
    //     binary_op: Option<fn(i32, &mut f32, &f32, &f32)>,
    // ) -> Tensor {
    //     // TODO: check type f32
    //     Tensor {
    //         tensor: unsafe {
    //             ffi_ggml::ggml_map_binary_f32(self.ctx, a.tensor, b.tensor, binary_op)
    //         },
    //     }
    // }

    pub fn view_1d(&self, a: &Tensor, nelements: usize, offset: usize) -> Result<Tensor, ()> {
        // TODO: check nelements is within bounds
        // TODO: check offset is ok
        // TODO: check a has the right dimension
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_view_1d(
                    self.ctx,
                    a.tensor,
                    nelements.try_into().unwrap(),
                    offset.try_into().unwrap(),
                )
            },
        })
    }

    pub fn view_2d(
        &self,
        a: &Tensor,
        nelements0: usize,
        nelements1: usize,
        nbytes1: usize,
        offset: usize,
    ) -> Result<Tensor, ()> {
        // TODO: check nelements is within bounds
        // TODO: check offset is ok
        // TODO: check a has the right dimension
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_view_2d(
                    self.ctx,
                    a.tensor,
                    nelements0.try_into().unwrap(),
                    nelements1.try_into().unwrap(),
                    nbytes1.try_into().unwrap(),
                    offset.try_into().unwrap(),
                )
            },
        })
    }

    pub fn view_3d(
        &self,
        a: &Tensor,
        nelements0: usize,
        nelements1: usize,
        nelements2: usize,
        nbytes1: usize,
        nbytes2: usize,
        offset: usize,
    ) -> Result<Tensor, ()> {
        // TODO: check nelements is within bounds
        // TODO: check offset is ok
        // TODO: check a has the right dimension
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_view_3d(
                    self.ctx,
                    a.tensor,
                    nelements0.try_into().unwrap(),
                    nelements1.try_into().unwrap(),
                    nelements2.try_into().unwrap(),
                    nbytes1.try_into().unwrap(),
                    nbytes2.try_into().unwrap(),
                    offset.try_into().unwrap(),
                )
            },
        })
    }

    pub fn reshape_2d(
        &self,
        a: &Tensor,
        nelements0: usize,
        nelements1: usize,
    ) -> Result<Tensor, ()> {
        // TODO: check on nelements0 and nelements1
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_reshape_2d(
                    self.ctx,
                    a.tensor,
                    nelements0.try_into().unwrap(),
                    nelements1.try_into().unwrap(),
                )
            },
        })
    }

    pub fn reshape_3d(
        &self,
        a: &Tensor,
        nelements0: usize,
        nelements1: usize,
        nelements2: usize,
    ) -> Result<Tensor, ()> {
        // TODO: check on nelements0 and nelements1 and nelements2
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_reshape_3d(
                    self.ctx,
                    a.tensor,
                    nelements0.try_into().unwrap(),
                    nelements1.try_into().unwrap(),
                    nelements2.try_into().unwrap(),
                )
            },
        })
    }

    pub fn permute(
        &self,
        a: &Tensor,
        axis0: usize,
        axis1: usize,
        axis2: usize,
        axis3: usize,
    ) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_permute(
                    self.ctx,
                    a.tensor,
                    axis0.try_into().unwrap(),
                    axis1.try_into().unwrap(),
                    axis2.try_into().unwrap(),
                    axis3.try_into().unwrap(),
                )
            },
        })
    }

    pub fn diag_mask_inf(&self, a: &Tensor, n_past: usize) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_diag_mask_inf(self.ctx, a.tensor, n_past.try_into().unwrap())
            },
        })
    }

    pub fn rope(&self, a: &Tensor, n_past: usize, n_dims: usize, skip: bool) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_rope(
                    self.ctx,
                    a.tensor,
                    n_past.try_into().unwrap(),
                    n_dims.try_into().unwrap(),
                    skip.try_into().unwrap(),
                )
            },
        })
    }

    pub fn flash_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        masked: bool,
    ) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_flash_attn(self.ctx, query.tensor, key.tensor, value.tensor, masked)
            },
        })
    }

    pub fn flash_feed_forward(
        &self,
        a: &Tensor,
        b0: &Tensor,
        b1: &Tensor,
        c0: &Tensor,
        c1: &Tensor,
    ) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe {
                ffi_ggml::ggml_flash_ff(
                    self.ctx, a.tensor, b0.tensor, b1.tensor, c0.tensor, c1.tensor,
                )
            },
        })
    }

    // TODO: map unary, map binary
    // TODO: copy via the copy trait
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe { ffi_ggml::ggml_free(self.ctx) }
    }
}

pub struct CGraph {
    graph: ffi_ggml::ggml_cgraph,
}

impl CGraph {
    pub fn build_forward(tensor: &Tensor) -> Self {
        Self {
            graph: unsafe { ffi_ggml::ggml_build_forward(tensor.tensor) },
        }
    }

    pub fn build_backward(ctx: &Context, graph: &mut CGraph, keep: bool) -> Self {
        Self {
            graph: unsafe { ffi_ggml::ggml_build_backward(ctx.ctx, &mut graph.graph, keep) },
        }
    }

    pub fn build_forward_expand(&mut self, tensor: &Tensor) {
        unsafe { ffi_ggml::ggml_build_forward_expand(&mut self.graph, tensor.tensor) }
    }

    pub fn compute(&mut self, ctx: &Context) {
        unsafe { ffi_ggml::ggml_graph_compute(ctx.ctx, &mut self.graph) }
    }

    pub fn reset(&mut self) {
        unsafe { ffi_ggml::ggml_graph_reset(&mut self.graph) }
    }

    // TODO: implement display trait
    pub fn print(&self) {
        unsafe { ffi_ggml::ggml_graph_print(&self.graph) }
    }
}

#[derive(Debug)]
pub struct Tensor {
    tensor: *mut ffi_ggml::ggml_tensor,
}

impl Tensor {
    pub fn nbytes(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_nbytes(self.tensor) };
        res.try_into().unwrap()
    }

    pub fn nelements(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_nelements(self.tensor) };
        res.try_into().unwrap()
    }

    pub fn element_size(&self) -> usize {
        let res = unsafe { ffi_ggml::ggml_element_size(self.tensor) };
        res.try_into().unwrap()
    }

    // TODO: make get_data and get_data_1d generic over the tensor type
    pub fn get_data_f32(&self) -> Result<Vec<f32>, ()> {
        let ptr = unsafe { ffi_ggml::ggml_get_data_f32(self.tensor) };
        match ptr.is_null() {
            true => Err(()),
            false => {
                let length = self.nelements();
                let capacity = self.nbytes();
                let data = unsafe { Vec::from_raw_parts(ptr, length, capacity) };
                Ok(data)
            }
        }
    }

    pub fn get_data_f32_1d(&self, idx: usize) -> Result<f32, ()> {
        match idx >= self.nelements() {
            true => Err(()),
            false => Ok(unsafe { ffi_ggml::ggml_get_f32_1d(self.tensor, idx.try_into().unwrap()) }),
        }
    }

    // TODO: make set_data and set_data_1d generic over the tensor type
    pub fn set_data_f32(&self, value: f32) -> Tensor {
        Tensor {
            tensor: unsafe { ffi_ggml::ggml_set_f32(self.tensor, value) },
        }
    }

    pub fn set_data_f32_1d(&self, value: f32, idx: usize) -> Result<(), ()> {
        match idx >= self.nelements() {
            true => Err(()),
            false => Ok(unsafe {
                ffi_ggml::ggml_set_f32_1d(self.tensor, idx.try_into().unwrap(), value)
            }),
        }
    }

    pub fn set_zero(&self) -> Tensor {
        Tensor {
            tensor: unsafe { ffi_ggml::ggml_set_zero(self.tensor) },
        }
    }
}

#[cfg(test)]
mod test;
