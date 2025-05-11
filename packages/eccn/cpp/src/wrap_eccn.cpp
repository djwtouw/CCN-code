#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "cc_sequential.h"
#include "cc_network.h"


PYBIND11_MODULE(_eccn, m)
{
    m.def("ccs_logistic", &ccs_logistic, "Test");
    m.def("ccn_logistic", &ccn_logistic, "Test");
}
