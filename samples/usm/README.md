# Unified Shared Memory Samples

This directory contains samples demonstrating Unified Shared Memory (USM).
Unified Shared Memory is intended to bring pointer-based programming to SYCL.

## Unified Shared Memory Status

USM is currently a SYCL extension, supported in Data Parallel C++ (DPC++).
The latest draft of the Unified Shared Memory extension specification can be found [here](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc).

These samples were tested with DPC++ Beta 5.

## Unified Shared Memory Advantages

Unified Shared Memory (USM) provides:

* Easier integration into existing code bases by representing memory allocations as pointers rather than handles, with full support for pointer arithmetic into allocations.

* Fine-grain control over ownership and accessibility of memory allocations, to optimally choose between performance and programmer convenience.

* A simpler programming model, by automatically migrating some memory allocations between devices and the host.

## Summary of Unified Shared Memory Samples

* [usmqueries](./00_usmqueries): Queries and prints the USM capabilities of a device.
* [dmemhelloworld](./100_dmemhelloworld): Copy one "device" memory allocation to another.
* [hmemhelloworld](./200_hmemhelloworld): Copy one "host" memory allocation to another.
* [smemhelloworld](./300_smemhelloworld): Copy one "shared" memory allocation to another.

These samples are closely derived from corresponding OpenCL [USM Samples](https://github.com/bashbaug/SimpleOpenCLSamples/tree/master/samples/usm).
