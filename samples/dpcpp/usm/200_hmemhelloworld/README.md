# hmemhelloworld

## Sample Purpose

This sample demonstrates usage of host memory allocations.
Other similar samples demonstrate usage of device memory and shared memory allocations.

Host memory allocations are owned by the host, and generally trade wide access for potentially lower performance.
Because of its wide access, using host memory is one of the easiest ways to enable an application to use Unified Shared Memory, albeit at a potential performance cost.

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates host memory using `sycl::malloc_host` and frees it using `sycl::free`.

Since host memory may be directly accessed and manipulated on the host, this sample does not need to use any special Unified Shared Memory APIs to copy to or from a host allocation, or to map or unmap a host allocation.
Instead, this sample simply ensures that copy kernel is complete before verifying that the copy was performed correctly.
For simplicity, this sample ensures all commands are complete `wait` on the queue, but other completion mechanisms could be used instead that may be more efficient.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an accessor.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the SYCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the SYCL platform to execute the sample on.
