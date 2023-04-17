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

    pub fn add(&self, a: &Tensor, b: &Tensor) -> Tensor {
        Tensor {
            tensor: unsafe { ffi_ggml::ggml_add(self.ctx, a.tensor, b.tensor) },
        }
    }
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
    pub fn duplicate(&self, ctx: &Context) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe { ffi_ggml::ggml_dup_tensor(ctx.ctx, self.tensor) },
        })
    }

    pub fn view(&self, ctx: &Context) -> Result<Tensor, ()> {
        Ok(Tensor {
            tensor: unsafe { ffi_ggml::ggml_view_tensor(ctx.ctx, self.tensor) },
        })
    }

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
    pub fn set_data_f32(&self, value: f32) {
        let _ = unsafe { ffi_ggml::ggml_set_f32(self.tensor, value) };
    }

    pub fn set_data_f32_1d(&self, value: f32, idx: usize) -> Result<(), ()> {
        match idx >= self.nelements() {
            true => Err(()),
            false => Ok(unsafe {
                ffi_ggml::ggml_set_f32_1d(self.tensor, idx.try_into().unwrap(), value)
            }),
        }
    }
}

//     struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);

//     */
// }

// /*
//     functions
//     helpers
//     ggml_cpy

//     Ops
//     ggml_norm
//     ggml_add
//     ggml_mul
//     ggml_repeat
//     ggml_rope
//     ggml_reshape_3d
//     ggml_mul_mat
//     ggml_transpose
//     ggml_view_1d
//     ggml_view_2d
//     ggml_permute
//     ggml_scale
//     ggml_soft_max

// */
#[cfg(test)]
mod test;
