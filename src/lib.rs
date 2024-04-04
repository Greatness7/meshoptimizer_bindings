use pyo3::{exceptions::PyValueError, prelude::*};

use numpy::{prelude::*, PyArray1, PyArray2};

use meshopt::ffi::{
    meshopt_optimizeOverdraw, meshopt_optimizeVertexCache, meshopt_optimizeVertexFetchRemap,
    meshopt_remapIndexBuffer,
};

use std::os::raw::{c_float, c_uint};

type Vertices<'py> = Bound<'py, PyArray2<c_float>>;
type Triangles<'py> = Bound<'py, PyArray2<c_uint>>;
type IndexRemap<'py> = Bound<'py, PyArray1<c_uint>>;

#[pyfunction]
fn optimize<'py>(
    py: Python<'py>,
    vertices: Vertices<'py>,
    triangles: Triangles<'py>,
) -> PyResult<(IndexRemap<'py>, Triangles<'py>)> {
    if !vertices.is_c_contiguous() || !triangles.is_c_contiguous() {
        return Err(PyValueError::new_err("Inputs must be C-contiguous"));
    }

    let vertex_count = vertices.shape()[0] as usize;
    let vertex_stride = vertices.strides()[0] as usize;

    unsafe {
        let vertex_remap = PyArray1::new_bound(py, vertex_count, false);
        let triangles_opt = PyArray2::new_bound(py, triangles.dims(), false);

        meshopt_optimizeVertexCache(
            triangles_opt.data(),
            triangles.data(),
            triangles.len(),
            vertex_count,
        );
        meshopt_optimizeOverdraw(
            triangles_opt.data(),
            triangles_opt.data(),
            triangles.len(),
            vertices.data(),
            vertex_count,
            vertex_stride,
            1.05,
        );
        meshopt_optimizeVertexFetchRemap(
            vertex_remap.data(),
            triangles_opt.data(),
            triangles.len(),
            vertex_count,
        );
        meshopt_remapIndexBuffer(
            triangles_opt.data(),
            triangles_opt.data(),
            triangles.len(),
            vertex_remap.data(),
        );

        Ok((vertex_remap, triangles_opt))
    }
}

#[pymodule]
fn meshoptimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(optimize, m)?)?;
    Ok(())
}
