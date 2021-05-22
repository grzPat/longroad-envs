#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include "world.hpp"
namespace py = pybind11;

namespace longroad{
PYBIND11_MODULE(world, m) {
    py::class_<World_i>(m, "World_i")
        .def(py::init<int,bool,bool,double,double>())
        .def("step", &World_i::step)
        .def("seed", &World_i::setSeed)
        .def("reset", &World_i::reset)
        .def("last_states", &World_i::lastStates, py::return_value_policy::reference_internal)
        .def("last_rewards", &World_i::lastRewards, py::return_value_policy::reference_internal)
        .def("avgtime", &World_i::avgTime)
        ;

}
}