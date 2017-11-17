#include "Differentiation.h"

#include <Eigen/Dense>

MatrixXd VerticalDerivativeMatrix(BoundaryCondition originalBC, double L, int N)
{
    MatrixXd D = MatrixXd::Zero(N,N);

    for (int j=0; j<N; j++)
    {
        if(j==0)
        {
            D(j,j+2) = j+2;
        }
        else if(j==1)
        {
            // use (anti)symmetry of basis function to replace missing mode
            if (originalBC==BoundaryCondition::Neumann)
            {
                D(j, j) = -3*j;
            }
            else
            {
                D(j, j) = -1*j;
            }
            D(j, j+2) = j+2;
        }
        else
        {
            D(j, j-2) = j-2;
            D(j,j) = -2*j;

            if(j+2 < N) // we have lost some terms here
            {
                D(j,j+2) = j+2;
            }
        }
    }

    D /= (4*L);
    if (originalBC==BoundaryCondition::Neumann)
    {
        D = -D;
    }

    D.row(0) *= 2;
    D.col(0) /= 2;
    D.row(N-1) *= 2;
    D.col(N-1) /= 2;

    return D;
}


MatrixXd VerticalSecondDerivativeMatrix(BoundaryCondition bc, double L, int N)
{
    if (bc == BoundaryCondition::Neumann)
    {
        return VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L, N)
                * VerticalDerivativeMatrix(BoundaryCondition::Neumann, L, N);
    }
    else
    {
        return VerticalDerivativeMatrix(BoundaryCondition::Neumann, L, N)
                * VerticalDerivativeMatrix(BoundaryCondition::Dirichlet, L, N);
    }
}


ArrayXd k(int n)
{
    if (n==1)
    {
        // handle this separately for 2D
        return ArrayXd::Zero(1);
    }
    assert(n % 2 == 0); // odd case not handled
    assert(n > 0);

    ArrayXd k(n);

    // using this for k gives a result which matches the FT of the real
    // derivative
    k << ArrayXd::LinSpaced(n / 2, 0, n / 2 - 1),
         ArrayXd::LinSpaced(n / 2, -n / 2, -1);

    return k;
}

DiagonalMatrix<double, -1> FourierSecondDerivativeMatrix(double L, int N)
{
    VectorXd ret = -4*pi*pi*k(N)*k(N)/(L*L);
    return ret.asDiagonal();
}

DiagonalMatrix<complex, -1> FourierDerivativeMatrix(double L, int N)
{
    VectorXcd ret = 2.0*pi*i*k(N)/L;
    return ret.asDiagonal();
}
