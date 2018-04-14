#pragma once

#include "Stratiflow.h"

MatrixXc OrrSommerfeldLHS(stratifloat k);
MatrixXc OrrSommerfeldRHS(stratifloat k);

ArrayXc CalculateEigenvalues(stratifloat k,
                             MatrixXc *w_eigen = nullptr,
                             MatrixXc *b_eigen = nullptr);

stratifloat LargestGrowth(stratifloat k,
                          Field1D<complex, N1, N2, N3>* w=nullptr,
                          Field1D<complex, N1, N2, N3>* b=nullptr,
                          stratifloat* imag=nullptr);

void EigenModes(stratifloat k, DirichletModal& u1, DirichletModal& u2, DirichletModal& u3, DirichletModal& b);