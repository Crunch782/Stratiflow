#include "OrrSommerfeld.h"
#include "Graph.h"
#include "Differentiation.h"

int main(int argc, char *argv[])
{
    Ri = std::stof(argv[1]);

    stratifloat kmax;
    stratifloat growthmax = -10000;

    // the largest wavelength we want to consider
    stratifloat lambdamax = 14.0;
    stratifloat lambdamin = 10.0f;
    stratifloat kmin = 2*pi/lambdamax;

    stratifloat k_lower = kmin;
    stratifloat k_upper = 2*pi/lambdamin;

    for (int n=0; n<6; n++)
    {
        stratifloat deltak = (k_upper-k_lower)/10;

        for (stratifloat k=k_lower; k<=k_upper; k+=deltak)
        {
            auto largest = LargestGrowth(k);

            std::cout << k << " : " << largest << std::endl;

            if (largest>growthmax)
            {
                growthmax = largest;
                kmax = k;
            }
        }

        k_lower = std::max(kmax - deltak, kmin);
        k_upper = kmax + deltak;
    }

    std::cout << "Maximum growth rate " << growthmax << " at " << kmax << std::endl;
    std::cout << "Wavelength of fastest growing mode is " << 2*pi/kmax << std::endl;

    NormalModal u;
    NormalModal v;
    StaggeredModal w;
    NormalModal b;

    EigenModes(kmax, u, v, w, b);

    HeatPlot(u, L1, L3, 0, "u_eig.png");
    HeatPlot(v, L1, L3, 0, "v_eig.png");
    HeatPlot(w, L1, L3, 0, "w_eig.png");
    HeatPlot(b, L1, L3, 0, "b_eig.png");


    return 0;
}