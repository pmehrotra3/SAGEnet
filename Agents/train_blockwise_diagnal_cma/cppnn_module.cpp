#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Easiest hack: include your implementation directly
#include "Feedforward.cpp"

namespace py = pybind11;

PYBIND11_MODULE(cppnn, m) {
    py::class_<neural_network>(m, "NeuralNetwork")
        .def(py::init<int, const std::vector<int>&, int>(),
             py::arg("input_size"), py::arg("hidden_layer_sizes"), py::arg("output_size"))
        .def("forward", &neural_network::forward)
        .def("get_param", &neural_network::get_param)
        .def("set_param", &neural_network::set_param);
}

