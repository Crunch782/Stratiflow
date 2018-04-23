#include "IMEXRK.h"
#include "OSUtils.cpp"
#include "OrrSommerfeld.h"

int main(int argc, char *argv[])
{
    DumpParameters();

    stratifloat targetTime = 400.0;
    stratifloat integrateTarget = 40.0;
    stratifloat energy = 0.001;

    f3_init_threads();
    f3_plan_with_nthreads(omp_get_max_threads());

    std::cout << "Creating solver..." << std::endl;
    IMEXRK solver;

    if (argc == 2)
    {
        std::cout << "Loading ICs..." << std::endl;
        solver.LoadFlow(argv[1]);
    }
    else
    {
        std::cout << "Setting ICs..." << std::endl;
        NormalNodal   initialU1;
        NormalNodal   initialU2;
        StaggeredNodal initialU3;
        NormalNodal initialB;
        auto x3 = VerticalPoints(L3, N3);


        NormalModal initialu1;
        NormalModal initialu2;
        StaggeredModal initialu3;
        NormalModal initialb;

        std::cout << "Calculating Eigenmode..." << std::endl;

        // // find which mode to use
        // int mode = 0;
        // stratifloat growth = -1000;
        // for (int m=1; m<5; m++)
        // {
        //     stratifloat sigma = LargestGrowth(2*pi*m/L1);
        //     if (sigma > growth)
        //     {
        //         growth = sigma;
        //         mode = m;
        //     }
        // }

        // // add the eigenmode
        // EigenModes(2*pi*mode/L1, initialu1, initialu2, initialu3, initialb);
        // initialu1.ToNodal(initialU1);
        // initialu2.ToNodal(initialU2);
        // initialu3.ToNodal(initialU3);
        // initialb.ToNodal(initialB);

        // add a perturbation to allow instabilities to develop
        stratifloat bandmax = 4;
        for (int j=0; j<N3; j++)
        {
            if (x3(j) > -bandmax && x3(j) < bandmax)
            {
                initialU1.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, N1, N2>::Random(N1, N2);
                initialU2.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, N1, N2>::Random(N1, N2);
                initialU3.slice(j) += 0.01*(bandmax*bandmax-x3(j)*x3(j))
                    * Array<stratifloat, N1, N2>::Random(N1, N2);
            }
        }
        solver.SetInitial(initialU1, initialU2, initialU3, initialB);
    }

    std::ofstream energyFile("flow_stats.dat");

    // add background flow
    std::cout << "Setting background..." << std::endl;
    solver.SetBackground(InitialU, InitialB);

    if (argc == 1)
    {
        solver.FilterAll();
        solver.PopulateNodalVariables();
        solver.RemoveDivergence(0.0f);
        solver.RescaleForEnergy(energy);
        solver.SolveForPressure();
    }

    stratifloat totalTime = 0.0f;

    stratifloat saveEvery = 5.0f;
    int lastFrame = -1;
    int step = 0;

    std::cout << "E0: " << solver.KE() + solver.PE() << std::endl;

    StaggeredModal wIntegrated;
    stratifloat w2Integrated = 0;

    solver.PrepareRun("images/");
    solver.PlotAll(std::to_string(totalTime)+".png", true);
    while (totalTime < targetTime)
    {
        solver.TimeStep();
        totalTime += solver.deltaT;

        if(step%50==0)
        {
            stratifloat cfl = solver.CFL();
            std::cout << "  Step " << step << ", time " << totalTime
                    << ", CFL number: " << cfl << std::endl;

            std::cout << "  Average timings: " << solver.totalExplicit / (step+1)
                    << ", " << solver.totalImplicit / (step+1)
                    << ", " << solver.totalDivergence / (step+1)
                    << std::endl;
        }

        if (totalTime < integrateTarget)
        {
            wIntegrated += solver.deltaT * solver.u3;
            w2Integrated += solver.deltaT * InnerProd(solver.u3, solver.u3, L3);
        }

        int frame = static_cast<int>(totalTime / saveEvery);

        if (frame>lastFrame)
        {
            lastFrame=frame;

            solver.PlotAll(std::to_string(totalTime)+".png", true);
            solver.SaveFlow("snapshots/"+std::to_string(totalTime)+".fields");

            energyFile << totalTime
                    << " " << solver.KE()
                    << " " << solver.PE()
                    << std::endl;
        }

        step++;

    }

    std::cout << InnerProd(wIntegrated, wIntegrated, L3)/w2Integrated/integrateTarget << std::endl;


    f3_cleanup_threads();

    return 0;
}
