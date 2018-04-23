#pragma once
#include "Field.h"
#include "Parameters.h"
#include "Differentiation.h"

constexpr int M1 = N1/2 + 1;

class NormalNodal : public NodalField<N1,N2,N3>
{
public:
    NormalNodal() : NodalField(GridType::Normal) {}
    using NodalField::operator=;
};

class NormalModal : public ModalField<N1,N2,N3>
{
public:
    NormalModal() : ModalField(GridType::Normal) {}
    using ModalField::operator=;
};

class StaggeredNodal : public NodalField<N1,N2,N3>
{
public:
    StaggeredNodal() : NodalField(GridType::Staggered) {}
    using NodalField::operator=;

};

class StaggeredModal : public ModalField<N1,N2,N3>
{
public:
    StaggeredModal() : ModalField(GridType::Staggered) {}
    using ModalField::operator=;
};

class Normal1D : public Nodal1D<N1,N2,N3>
{
public:
    Normal1D() : Nodal1D(GridType::Normal) {}
    using Nodal1D::operator=;
};

class Staggered1D : public Nodal1D<N1,N2,N3>
{
public:
    Staggered1D() : Nodal1D(GridType::Staggered) {}
    using Nodal1D::operator=;
};

template<typename T>
Dim1MatMul<T, complex, complex, M1, N2, N3> ddx(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim1Derivative = FourierDerivativeMatrix(L1, N1, 1);

    return Dim1MatMul<T, complex, complex, M1, N2, N3>(dim1Derivative, f);
}

template<typename T>
Dim2MatMul<T, complex, complex, M1, N2, N3> ddy(const StackContainer<T, complex, M1, N2, N3>& f)
{
    static DiagonalMatrix<complex, -1> dim2Derivative = FourierDerivativeMatrix(L2, N2, 2);

    return Dim2MatMul<T, complex, complex, M1, N2, N3>(dim2Derivative, f);
}

template<typename A, typename T, int K1, int K2, int K3>
Dim3MatMul<A, stratifloat, T, K1, K2, K3> ddz(const StackContainer<A, T, K1, K2, K3>& f)
{
    if (f.Grid() == GridType::Normal)
    {
        static MatrixX dim3Derivative = VerticalDerivativeMatrix(L3, N3, f.Grid());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3Derivative, f, GridType::Staggered);
    }
    else
    {
        static MatrixX dim3Derivative = VerticalDerivativeMatrix(L3, N3, f.Grid());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(dim3Derivative, f, GridType::Normal);
    }
}

template<typename A, typename T, int K1, int K2, int K3>
Dim3MatMul<A, stratifloat, T, K1, K2, K3> Reinterpolate(const StackContainer<A, T, K1, K2, K3>& f)
{
    if (f.Grid() == GridType::Normal)
    {
        static MatrixX reint = VerticalReinterpolationMatrix(L3, N3, f.Grid());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(reint, f, GridType::Staggered);
    }
    else
    {
        static MatrixX reint = VerticalReinterpolationMatrix(L3, N3, f.Grid());
        return Dim3MatMul<A, stratifloat, T, K1, K2, K3>(reint, f, GridType::Normal);
    }
}

namespace
{
void InterpolateProduct(const NormalNodal& A, const NormalNodal& B, NormalModal& to)
{
    static NormalNodal prod;
    prod = A*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NormalNodal& A, const NormalNodal& B, StaggeredModal& to)
{
    static StaggeredNodal prod;
    prod = Reinterpolate(A*B);
    prod.ToModal(to);
}
void InterpolateProduct(const NormalNodal& A, const StaggeredNodal& B, StaggeredModal& to)
{
    static StaggeredNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const StaggeredNodal& B, const NormalNodal& A, StaggeredModal& to)
{
    static StaggeredNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NormalNodal& B, const StaggeredNodal& A, NormalModal& to)
{
    static NormalNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const StaggeredNodal& A, const NormalNodal& B, NormalModal& to)
{
    static NormalNodal prod;
    prod = Reinterpolate(A)*B;
    prod.ToModal(to);
}
void InterpolateProduct(const StaggeredNodal& A, const StaggeredNodal& B, NormalModal& to)
{
    static NormalNodal prod;
    prod = Reinterpolate(A*B);
    prod.ToModal(to);
}
void InterpolateProduct(const StaggeredNodal& A, const StaggeredNodal& B, StaggeredModal& to)
{
    static StaggeredNodal prod;
    prod = A*B;
    prod.ToModal(to);
}
void InterpolateProduct(const NormalNodal& A1, const NormalNodal& A2,
                        const NormalNodal& B1, const NormalNodal& B2,
                        NormalModal& to)
{
    static NormalNodal prod;
    prod = A1*B1 + A2*B2;
    prod.ToModal(to);
}
void InterpolateProduct(const NormalNodal& A1, const NormalNodal& A2,
                        const StaggeredNodal& B1, const StaggeredNodal& B2,
                        StaggeredModal& to)
{
    static StaggeredNodal prod;
    prod = Reinterpolate(A1)*B1 + Reinterpolate(A2)*B2;
    prod.ToModal(to);
}
}
