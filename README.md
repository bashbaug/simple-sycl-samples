# Simple SYCL<sup>TM</sup> Samples

This repo contains simple SYCL samples.

## Code Structure

```
README.md               This file
LICENSE                 License information
CMakeLists.txt          Top-level CMakefile
samples/                Samples
```

## How to Build the Samples

This section is a work in progress!

These samples are currently built on Windows using the Beta 5 DPC++ compiler.

I have had best luck building on Windows using a Ninja generator.
To build the samples:

1. Setup oneAPI variables:

    /path/to/inteloneapi/setvars.bat

2. Create build files using CMake, specifying the DPC++ toolchain.  For example:

    mkdir build && cd build
    cmake -G Ninja -DCMAKE_TOOLCHAIN_FILE=../dpcpp_toolchain.cmake ..

3. Build with the generated build files:

    ninja install

The files in the top-level `samples` directory are intended to be standard SYCL samples and should build and run on any SYCL implementation.

The files in the `dpcpp` directory require SYCL extensions and hence will only build and run with the DPC++ compiler.

## License

These samples are licensed under the [MIT License](LICENSE).

---
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.

\* Other names and brands may be claimed as the property of others.