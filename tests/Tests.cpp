#define CATCH_CONFIG_RUNNER
#include <catch.h>

#ifdef USE_OMP
#include <omp.h>
#endif

#include "FFT.h"
#include "Constants.h"


int main(int argc, char* argv[])
{
    int result = Catch::Session().run(argc, argv);

    Cleanup();
    return result;
}