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

These samples are currently built on Windows and Linux (Ubuntu 22.04) using the 2023.1 DPC++ compiler.
Building the samples requires CMake 3.10 or newer for Linux, or CMake 3.25 or newer for Windows.

I have been using the Ninja CMake generator.
Here is how I have been building the samples:

1. Setup oneAPI variables:

    For Windows:

    ```sh
    \path\to\inteloneapi\setvars.bat
    ```

    For Linux:

    ```sh
    source /path/to/inteloneapi/setvars.sh
    ```

2. Create build files using CMake.  For example:

    ```sh
    mkdir build && cd build
    cmake -G Ninja ..
    ```

3. Build with the generated build files:

    ```sh
    ninja install
    ```

The files in the top-level `samples` directory are intended to be standard SYCL samples and should build and run on any SYCL implementation.

The files in the `dpcpp` directory require SYCL extensions and hence will only build and run with the DPC++ compiler.

## License

These samples are licensed under the [MIT License](LICENSE).

---
OpenCL and the OpenCL logo are trademarks of Apple Inc. used by permission by Khronos.

\* Other names and brands may be claimed as the property of others.