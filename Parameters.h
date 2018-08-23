#pragma once
#include "Constants.h"
#include <string>

// These are the used-modifiable parameters for Stratiflow

// These are runtime parameters - values in Parameters.cpp
extern stratifloat L1;
extern stratifloat L2;
extern stratifloat L3;
extern stratifloat Re;
extern stratifloat Ri;
extern stratifloat R;
extern stratifloat Pr;
extern stratifloat Pe;
extern bool EnforceSymmetry;

// These must be defined at compile time

// SOLVER PARAMETERS //
 constexpr int N1 = 256; // Number of streamwise gridpoints
 constexpr int N2 = 64;   // Number of spanwise gridpoints
 constexpr int N3 = 256; // Number of vertical gridpoints

 constexpr bool ThreeDimensional = true; // whether to resolve spanwise direction
 constexpr bool EvolveBackground = true;

// background shear
inline stratifloat InitialU(stratifloat z)
{
    return tanh(z);
}

// background stratification
inline stratifloat InitialB(stratifloat z)
{
    return -tanh(R*z);
}

constexpr stratifloat zlim = 8;

inline stratifloat zFunc(stratifloat z)
{
    if (z>zlim || z<-zlim)
    {
        return 0;
    }
    else
    {
        return z;
    }
}

inline stratifloat zFilter(stratifloat z)
{
    if (z>zlim || z<-zlim)
    {
        return 0;
    }
    else
    {
        return 1;
    }
}

void DumpParameters();
void PrintParameters();
void LoadParameters(const std::string& file);
