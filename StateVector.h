#include "IMEXRK.h"

// This class contains a full state's information
// its operations are not particularly efficient
// so it should only be used for high level algorithms
class StateVector
{
public:
    NeumannModal u1;
    DirichletModal u3;
    NeumannModal b;
    NeumannModal p;

    void FullEvolve(stratifloat T, StateVector& result, bool snapshot = false) const;

    void LinearEvolve(stratifloat T, StateVector& result) const;

    void AdjointEvolve(stratifloat T, StateVector& result) const;

    void CalcPressure()
    {
        CopyToSolver();
        solver.SolveForPressure();
        CopyFromSolver();
    }

    const StateVector& operator+=(const StateVector& other)
    {
        u1 += other.u1;
        u3 += other.u3;
        b += other.b;
        CalcPressure();

        return *this;
    }

    const StateVector& operator-=(const StateVector& other)
    {
        u1 -= other.u1;
        u3 -= other.u3;
        b  -= other.b;
        CalcPressure();

        return *this;
    }

    const StateVector& MulAdd(stratifloat a, const StateVector& B)
    {
        u1 += a*B.u1;
        u3 += a*B.u3;
        b  += a*B.b;
        CalcPressure();

        return *this;
    }

    const StateVector& operator*=(stratifloat other)
    {
        u1 *= other;
        u3 *= other;
        b  *= other;
        CalcPressure();

        return *this;
    }

    stratifloat Dot(const StateVector& other) const
    {
        stratifloat prod = 0.5f*(InnerProd(u1, other.u1, L3)
                               + InnerProd(u3, other.u3, L3)
                               + Ri*InnerProd(b, other.b, L3)); // TODO: is this correct PE?
        return prod;
    }

    stratifloat Norm2() const
    {
        return Dot(*this);
    }

    stratifloat Energy() const
    {
        return 0.5*Norm2();
    }

    stratifloat Norm() const
    {
        return sqrt(Norm2());
    }

    void Zero()
    {
        u1.Zero();
        u3.Zero();
        b.Zero();
        p.Zero();
    }

    void Rescale(stratifloat energy);

    void Randomise(stratifloat energy)
    {
        u1.RandomizeCoefficients(0.3);
        u3.RandomizeCoefficients(0.3);
        b.RandomizeCoefficients(0.3);

        Rescale(energy);
    }

    void LoadFromFile(const std::string& filename)
    {
        solver.LoadFlow(filename);
        CopyFromSolver();
    }

private:
    void CopyToSolver() const
    {
        solver.u1 = u1;
        solver.u3 = u3;
        solver.b = b;
        solver.u2.Zero();
        solver.p = p;
    }

    void CopyFromSolver()
    {
        CopyFromSolver(*this);
    }

    void CopyFromSolver(StateVector& into) const
    {
        into.u1 = solver.u1;
        into.u3 = solver.u3;
        into.b = solver.b;
        into.p = solver.p;
    }

    static IMEXRK solver;
};

StateVector operator+(const StateVector& lhs, const StateVector& rhs);
StateVector operator-(const StateVector& lhs, const StateVector& rhs);
StateVector operator*(stratifloat scalar, const StateVector& vector);
StateVector operator*(const StateVector& vector, stratifloat scalar);