#pragma once

#include <complex>

#ifdef USE_MPI
#include <mpi.h>
#endif

#ifdef USE_DOUBLE
    #define stratifloat double
    #define notstratifloat float

    #define ArrayX    ArrayXd
    #define ArrayXc   ArrayXcd
    #define ArrayXX   ArrayXXd
    #define ArrayXXc  ArrayXXcd
    #define VectorX   VectorXd
    #define VectorXc  VectorXcd
    #define MatrixX   MatrixXd
    #define MatrixXc  MatrixXcd

    #define f3_init_threads         fftw_init_threads
    #define f3_plan_with_nthreads   fftw_plan_with_nthreads
    #define f3_cleanup_threads      fftw_cleanup_threads
    #define f3_r2r_kind             fftw_r2r_kind
    #define f3_execute              fftw_execute
    #define f3_destroy_plan         fftw_destroy_plan
    #define f3_plan_many_r2r        fftw_plan_many_r2r
    #define f3_plan_many_dft_r2c    fftw_plan_many_dft_r2c
    #define f3_plan_many_dft_c2r    fftw_plan_many_dft_c2r
    #define f3_plan_r2r_1d          fftw_plan_r2r_1d
    #define f3_plan_dft_1d          fftw_plan_dft_1d
    #define f3_plan_dft_r2c_3d      fftw_plan_dft_r2c_3d
    #define f3_plan_dft_c2r_3d      fftw_plan_dft_c2r_3d
    #define f3_mpi_plan_dft_r2c_3d  fftw_mpi_plan_dft_r2c_3d
    #define f3_mpi_plan_dft_c2r_3d  fftw_mpi_plan_dft_c2r_3d
    #define f3_mpi_plan_many_dft_r2c  fftw_mpi_plan_many_dft_r2c
    #define f3_mpi_plan_many_dft_c2r  fftw_mpi_plan_many_dft_c2r
    #define f3_complex              fftw_complex
    #define f3_plan                 fftw_plan
    #define f3_mpi_init             fftw_mpi_init
    #define f3_mpi_cleanup          fftw_mpi_cleanup
    #define f3_mpi_local_size_3d    fftw_mpi_local_size_3d

    #ifdef USE_MPI
        #define MPI_STRATIFLOAT MPI_DOUBLE
    #endif
#else
    #define stratifloat float
    #define notstratifloat double

    #define ArrayX    ArrayXf
    #define ArrayXc   ArrayXcf
    #define ArrayXX   ArrayXXf
    #define ArrayXXc  ArrayXXcf
    #define VectorX   VectorXf
    #define VectorXc  VectorXcf
    #define MatrixX   MatrixXf
    #define MatrixXc  MatrixXcf

    #define f3_init_threads         fftwf_init_threads
    #define f3_plan_with_nthreads   fftwf_plan_with_nthreads
    #define f3_cleanup_threads      fftwf_cleanup_threads
    #define f3_r2r_kind             fftwf_r2r_kind
    #define f3_execute              fftwf_execute
    #define f3_destroy_plan         fftwf_destroy_plan
    #define f3_plan_many_r2r        fftwf_plan_many_r2r
    #define f3_plan_many_dft_r2c    fftwf_plan_many_dft_r2c
    #define f3_plan_many_dft_c2r    fftwf_plan_many_dft_c2r
    #define f3_plan_r2r_1d          fftwf_plan_r2r_1d
    #define f3_plan_dft_1d          fftwf_plan_dft_1d
    #define f3_plan_dft_r2c_3d      fftwf_plan_dft_r2c_3d
    #define f3_plan_dft_c2r_3d      fftwf_plan_dft_c2r_3d
    #define f3_mpi_plan_dft_r2c_3d  fftwf_mpi_plan_dft_r2c_3d
    #define f3_mpi_plan_dft_c2r_3d  fftwf_mpi_plan_dft_c2r_3d
    #define f3_mpi_plan_many_dft_r2c  fftwf_mpi_plan_many_dft_r2c
    #define f3_mpi_plan_many_dft_c2r  fftwf_mpi_plan_many_dft_c2r
    #define f3_complex              fftwf_complex
    #define f3_plan                 fftwf_plan
    #define f3_mpi_init             fftwf_mpi_init
    #define f3_mpi_cleanup          fftwf_mpi_cleanup
    #define f3_mpi_local_size_3d    fftwf_mpi_local_size_3d

    #ifdef USE_MPI
        #define MPI_STRATIFLOAT MPI_FLOAT
    #endif
#endif

#ifdef USE_MPI
    #define if_root \
    { \
    int if_rank;\
    MPI_Comm_rank(MPI_COMM_WORLD, &if_rank); \
    if (if_rank==0)

    #define endif }
#else
    #define if_root \
    if (true)

    #define endif
#endif

constexpr stratifloat pi = 3.14159265358979;
constexpr std::complex<stratifloat> i(0, 1);
constexpr stratifloat phi = 1.61803398874989;

// this number should be tweaked depending on the cache size of the processor
constexpr int LoopBlockSize = 16;

// this is a (hopefully) cache efficient loop for transposes
#define for2D(n1,n3) \
for (int k3 = 0; k3 < n3; k3 += LoopBlockSize) { \
for (int k1 = 0; k1 < n1; k1 += LoopBlockSize) { \
for (int j3 = k3; j3 < std::min(n3, k3 + LoopBlockSize); j3++) { \
for (int j1 = k1; j1 < std::min(n1, k1 + LoopBlockSize); j1++)

#define endfor2D \
}}}

using complex = std::complex<stratifloat>;

enum class BoundaryCondition
{
    Dirichlet, // these are located on a grid of height N3-1 including boundaries
    Neumann    // these are located on a grid of height N3 past boundaries
};

template <typename T>
int sgn(T val)
{
    return (T(0) < val) - (val < T(0));
}