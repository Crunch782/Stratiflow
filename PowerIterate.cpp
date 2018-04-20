#include "StateVector.h"

int main(int argc, char* argv[])
{
    const stratifloat T = 10;

    Ri = std::stof(argv[2]);

    StateVector stationaryPoint;
    stationaryPoint.LoadFromFile(argv[1]);

    StateVector stationaryPointEnd;

    stationaryPoint.FullEvolve(T, stationaryPointEnd, false, false);

    StateVector b_k;
    b_k.Randomise(0.0001, true);
    StateVector b_k1;

    int iterations = 0;
    while (true)
    {
        // normalise
        b_k *= 1/(b_k.Norm());

        // power iterate
        b_k.LinearEvolve(T, stationaryPoint, stationaryPointEnd, b_k1);

        // largest eigenvalue (in magnitude) of exponential matrix
        stratifloat mu = b_k1.Dot(b_k);

        stratifloat eigenvalueGuess = log(mu)/T;

        iterations++;
        std::cout << "ITERATION " << iterations << ", growth rate: " << eigenvalueGuess << std::endl;

        // for next step
        b_k = b_k1;

    }
}