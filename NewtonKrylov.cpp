#include "StateVector.h"
#include "OrrSommerfeld.h"

class NewtonKrylov
{
public:
    // using the GMRES routine, perform Newton-Raphson iteration
    // x : an initial guess, also the result when finished
    void Run(StateVector& x)
    {
        MakeCleanDir("ICs");

        StateVector dx; // update, solve into here to save reallocating memory

        stratifloat bestResidual;

        StateVector xPrevious;

        stratifloat Delta = 0.1;
        int step = 0;
        while(true)
        {
            step++;

            x.SaveToFile("ICs/"+std::to_string(step)+".fields");

            // first nonlinearly evolve current state
            StateVector rhs; // = G-x
            x.FullEvolve(T, rhs, false);

            linearAboutStart = x;
            linearAboutEnd = rhs;

            rhs -= x;

            stratifloat residual = rhs.Norm();

            std::cout << "NEWTON STEP " << step << ", RESIDUAL: " << residual << std::endl;

            if (residual < bestResidual || step == 1)
            {
                bestResidual = residual;
                xPrevious = x;
            }
            else
            {
                // not good enough, reduce trust region size and retry
                Delta /= 2;
                std::cout << "Delta: " << Delta << std::endl;

                // reset snapshots to previous
                // the time spent on this is small compared to GMRES time
                x = xPrevious;
                x.FullEvolve(T, rhs, true);
                rhs -= x;
            }

            // solve matrix system
            GMRES(rhs, dx, 0.01, Delta);

            // update
            x += dx;
        }
    }

private:
    // solves A x = G-x0 for x
    // where A = I-G_x
    // GMRES is a Krylov-subspace method, hence Newton-Krylov
    // Delta is a maximum size for x in the least squares solution
    void GMRES(const StateVector& rhs, StateVector& x, stratifloat epsilon, stratifloat Delta=0) const
    {
        int K = 512; // max iterations

        std::vector<StateVector> q(K);

        MatrixX H(K, K-1); // upper Hessenberg matrix representing A
        H.setZero();

        VectorX y; // result in new basis


        q[0] = rhs;
        q[0].EnforceBCs();

        stratifloat beta = q[0].Norm();
        std::cout << beta << std::endl;
        q[0] *= 1/beta;
        for (int k=1; k<K; k++)
        {
            // Arnoldi Algorithm
            // find orthogonal basis q1,...,qn
            // from x, A x, A^2 x, ...

            // q_k = A q_k-1
            StateVector Gq;
            q[k-1].LinearEvolve(T, linearAboutStart, linearAboutEnd, Gq);
            q[k] = q[k-1];
            q[k] -= Gq;

            // remove component in direction of preceding vectors
            for (int j=0; j<k; j++)
            {
                H(j,k-1) = q[j].Dot(q[k]);
                q[k].MulAdd(-H(j,k-1), q[j]);
            }

            // normalise
            H(k,k-1) = q[k].Norm();
            q[k] *= 1/H(k,k-1);

            // enforce BCs
            q[k].EnforceBCs();

            VectorX Beta(k+1);
            Beta.setZero();
            Beta[0] = beta;

            MatrixX subH = H.block(0,0,k+1,k);

            // Now we solve Hy = Beta

            // follows notation of Chandler & Kerswell 2013

            // first H = UDV*
            JacobiSVD<MatrixX> svd(subH, ComputeFullU | ComputeFullV);
            MatrixX U = svd.matrixU();
            MatrixX V = svd.matrixV();
            ArrayX d = svd.singularValues();

            // solve problem in space of singular vectors
            VectorX p = U.transpose() * Beta; // p = U* Beta
            VectorX z = p.array()/d; // D z = p

            // enforce trust region
            stratifloat mu = 0;
            while (z.norm() > Delta)
            {
                mu += 0.00001;

                for (int j=0; j<z.size(); j++)
                {
                    z(j) = p(j)*d(j)/(d(j)*d(j)+mu);
                }
            }

            std::cout << "For |z|<Delta, mu=" << mu << std::endl;

            // z = V* y
            y = V*z;


            stratifloat residual = (subH*y - Beta).norm()/beta;

            std::cout << "GMRES STEP " << k << ", RESIDUAL: " << residual << std::endl;

            if (residual < epsilon)
            {
                K = k+1;
                break;
            }
        }

        // Now compute the solution using the basis vectors
        x.Zero();
        for (int k=0; k<K-1; k++)
        {
            x.MulAdd(y[k], q[k]);
        }
    }

    stratifloat T = 5; // time interval for integration

    StateVector linearAboutStart;
    StateVector linearAboutEnd;
};

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cout << "Usage: 1.fields 2.fields Ri_1 Ri_2 Ri_new" << std::endl;
        return 1;
    }

    Ri = std::stof(argv[5]);

    DumpParameters();

    NewtonKrylov solver;

    StateVector field1;
    StateVector field2;
    field1.LoadFromFile(argv[1]);
    field2.LoadFromFile(argv[2]);

    stratifloat Ri_1 = std::stof(argv[3]);
    stratifloat Ri_2 = std::stof(argv[4]);

    StateVector guess = field2;

    if (Ri_1 != Ri_2)
    {
        StateVector derivative = field2;
        derivative -= field1;
        derivative *= 1/(Ri_2 - Ri_1);

        guess.MulAdd(Ri-Ri_2, derivative);
    }

    solver.Run(guess);
}
