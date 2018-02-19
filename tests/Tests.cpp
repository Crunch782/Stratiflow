#define CATCH_CONFIG_RUNNER
#include <catch.h>
#include <omp.h>

#include "FFT.h"
#include "Constants.h"


int main(int argc, char* argv[])
{
    int result = Catch::Session().run(argc, argv);

    Cleanup();
    return result;
}