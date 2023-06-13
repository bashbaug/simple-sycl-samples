# dmemhelloworld

## Sample Purpose

This is the first Unified Shared Memory sample that meaningfully stores and uses data in a Unified Shared Memory allocation.

This sample demonstrates usage of device memory allocations.
Other similar samples demonstrate usage of host memory and shared memory allocations.
Device memory allocations are owned by a specific device, and generally trade off high performance for limited access.
Kernels operating on device memory should perform just as well, if not better, than SYCL buffers / accessors.

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates device memory using `sycl::malloc_device` and frees it using `sycl::free`.

Since device memory cannot be directly accessed by the host, this sample initializes the source buffer by copying into it using `memcpy`.
This sample also uses `memcpy` to copy out of the destination buffer to verify that the copy was performed correctly.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an accessor.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the SYCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the SYCL platform to execute the sample on.
