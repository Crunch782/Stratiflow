# Stratiflow

Stratiflow is a package for the direct numerical simulation (DNS) of stratified shear flows.

## Compilation and Usage

### Prerequisites
* CMake, for building
* A C++ compiler supporting C++14
* FFTW
* Eigen 3
* Python with matplotlib for figures

Stratiflow has only been tested on Linux, but it should be straightforward to port to Windows and hopefully works out-of-the-box on macOS.

To build Stratiflow in any directory, run:
```
cmake -DCMAKE_BUILD_TYPE=Release <path to stratiflow git repository>
make
```

To check everything is working correctly, run the unit tests:
```
./tests/tests
```

Then to run Stratiflow:
```
./DNS
```

To run the direct-adjoint looping:
```
./DAL
```

### Acceleration
The fourier transforms in Stratiflow are the bottleneck, and then can be accelerated in two ways:
1. With MKL and an Intel CPU (may also work on AMD)

After installing MKL - which requires a license - and loading the environment, run `cmake` with the `-DMKL=On` option, and then build. Even if using CUDA too, this will still accelerate some linear algebra.

2. With CUDA and an Nvidia GPU

After installing the CUDA toolkit, run `cmake` with the `-DCUDA=On` option, and then build.

## How Stratiflow works

Stratiflow is designed specifically for unbounded stratified shear flows. The prototypical example would be for Kelvin-Helmholtz instability. It is possibly unique in that its domain is periodic in two (or one) horizontal dimensions, and infinite in the vertical.

This infinite domain is achieved by using [*Chebyshev rational functions*][1] as the basis functions in the vertical.

## Precision

By default, Stratiflow uses single precision floating point numbers, as increasing to double was not found to affect results but does impose a performance cost.
If for your problem, you require double precision, this case be enabled by using the `-DDOUBLE=On` flag when configuring with `cmake`.

[1]:http://www.springer.com/gb/book/9783540514879
