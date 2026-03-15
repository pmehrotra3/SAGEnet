// bindings.cpp
// Pybind11 bindings exposing the C++ neural_network class to Python as the module nn.
// Allows the CMA-ES optimiser (pycma) to set and get network parameters and run forward passes
// entirely in compiled C++, with automatic type conversion between Python lists and C++ vectors.
//
// Developed with assistance from:
//   Claude  (Anthropic)  — https://www.anthropic.com
//   ChatGPT (OpenAI)     — https://openai.com
//   Gemini  (Google)     — https://deepmind.google

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "Feedforward.cpp"

namespace py = pybind11;

PYBIND11_MODULE(nn, m) {
    py::class_<neural_network>(m, "NeuralNetwork")
        .def(py::init<int, const std::vector<int>&, int>(),
             py::arg("input_size"),
             py::arg("hidden_layer_sizes"),
             py::arg("output_size"))
        .def("forward",   &neural_network::forward)
        .def("get_param", &neural_network::get_param)
        .def("set_param", &neural_network::set_param);
}
