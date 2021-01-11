#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "meshoptimizer.h"

namespace py = pybind11;

py::tuple optimize(
	py::array_t<float, py::array::c_style | py::array::forcecast> vertices,
	py::array_t<unsigned int, py::array::c_style | py::array::forcecast> triangles
)
{
	auto vertex_count = vertices.shape(0);
	auto index_count = triangles.size();

	auto vertices_ptr = (float*)vertices.request().ptr;
	auto triangles_ptr = (unsigned int*)triangles.request().ptr;

	auto vertex_remap = py::array_t<unsigned int>(vertex_count);
	auto vertex_remap_ptr = (unsigned int*)vertex_remap.request().ptr;

	auto triangles_opt = py::array_t<unsigned int>(index_count);
	auto triangles_opt_ptr = (unsigned int*)triangles_opt.request().ptr;

	meshopt_optimizeVertexCache(triangles_opt_ptr, triangles_ptr, index_count, vertex_count);
	meshopt_optimizeOverdraw(triangles_opt_ptr, triangles_opt_ptr, index_count, &vertices_ptr[0], vertex_count, vertices.itemsize(), 1.05f);
	meshopt_optimizeVertexFetchRemap(vertex_remap_ptr, triangles_opt_ptr, index_count, vertex_count);
	meshopt_remapIndexBuffer(triangles_opt_ptr, triangles_opt_ptr, index_count, vertex_remap_ptr);

	triangles_opt.resize({triangles.shape(0), (ssize_t)3});

	return py::make_tuple(vertex_remap, triangles_opt);
}

PYBIND11_MODULE(meshoptimizer, m) {
	m.def("optimize", &optimize);
}
