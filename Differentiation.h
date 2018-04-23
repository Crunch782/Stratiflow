#pragma once

#include "Eigen.h"
#include "Constants.h"

// Computes a finite difference matrix for use with nodal forms
MatrixX VerticalSecondDerivativeMatrix(stratifloat L, int N, GridType originalGrid);
MatrixX VerticalDerivativeMatrix(stratifloat L, int N, GridType originalGrid);
MatrixX VerticalReinterpolationMatrix(stratifloat L, int N, GridType originalGrid);

DiagonalMatrix<stratifloat, -1> FourierSecondDerivativeMatrix(stratifloat L, int N, int dimension);
DiagonalMatrix<complex, -1> FourierDerivativeMatrix(stratifloat L, int N, int dimension);

template<typename M>
void Neumannify(M& matrix, GridType grid)
{
    int N3 = matrix.rows();

    // this only works for homogeneous bcs currently

    if (grid == GridType::Staggered)
    {
        // one sided derivatives, not so accurate
        matrix.coeffRef(0,0) = 1;
        matrix.coeffRef(0,1) = -1;
        matrix.coeffRef(0,2) = 0;

        matrix.coeffRef(N3-2, N3-3) = 1;
        matrix.coeffRef(N3-2, N3-2) = -1;
        matrix.coeffRef(N3-2, N3-4) = 0;

        matrix.coeffRef(N3-1, N3-2) = 0;
        matrix.coeffRef(N3-1, N3-1) = 1; // so non-singular
        matrix.coeffRef(N3-1, N3-3) = 0;
    }
    else
    {
        matrix.coeffRef(0,0) = 1;
        matrix.coeffRef(0,1) = -1;
        matrix.coeffRef(0,2) = 0;

        matrix.coeffRef(N3-1, N3-2) = 1;
        matrix.coeffRef(N3-1, N3-1) = -1;
        matrix.coeffRef(N3-1, N3-3) = 0;
    }
}

template<typename M>
void Dirichlify(M& matrix, GridType grid)
{
    int N3 = matrix.rows();

    if (grid == GridType::Staggered)
    {
        matrix.coeffRef(0,0) = 1;
        matrix.coeffRef(0,1) = 0;
        matrix.coeffRef(0,2) = 0;

        matrix.coeffRef(N3-1, N3-2) = 0;
        matrix.coeffRef(N3-1, N3-1) = 1;
        matrix.coeffRef(N3-1, N3-3) = 0;

        matrix.coeffRef(N3-2, N3-2) = 1;
        matrix.coeffRef(N3-2, N3-1) = 0;
        matrix.coeffRef(N3-2, N3-3) = 0;
        matrix.coeffRef(N3-2, N3-4) = 0;
    }
    else
    {
        // boundary is between these, so want average to be the value
        matrix.coeffRef(0,0) = 1;
        matrix.coeffRef(0,1) = 1;
        matrix.coeffRef(0,2) = 0;

        matrix.coeffRef(N3-1, N3-2) = 1;
        matrix.coeffRef(N3-1, N3-1) = 1;
        matrix.coeffRef(N3-1, N3-3) = 0;
    }
}
