#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(omigamax_core, m) {
    m.doc() = "omigamax C++ core";
}
