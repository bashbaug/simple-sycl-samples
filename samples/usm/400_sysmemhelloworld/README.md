# sysmemhelloworld

## Sample Purpose

This sample demonstrates usage of shared system memory allocations.
Shared system memory allocations are unique in that memory is allocated via a "system allocator", such as `malloc` or `free`, rather than via SYCL allocation functions.
Other similar samples demonstrate usage of SYCL memory allocations.

Like SYCL shared memory allocations, shared system memory allocations share ownership and are intended to implicitly migrate between the host and one or more devices.
Because shared memory allocations may migrate, they may generally be accessed with good performance on both the host and a device, albeit after paying for the cost of migration.
When supported, shared system memory is a great way to enable an application to use Unified Shared Memory with very good performance.

The sample initializes a source USM allocation, copies it to a destination USM allocation using a kernel, then checks on the host that the copy was performed correctly.

## Key APIs and Concepts

This sample allocates shared system memory using standard `malloc` and frees it using standard `free`.

Since shared memory may be directly accessed and manipulated on the host, this sample does not need to use any special Unified Shared Memory APIs to copy to or from a shared allocation, or to map or unmap a shared allocation.
Instead, this sample simply ensures that copy kernel is complete before verifying that the copy was performed correctly.
For simplicity, this sample ensures all commands are complete by calling `wait` on the queue, but other completion mechanisms could be used instead that may be more efficient.

Within a kernel, a Unified Shared Memory allocation can be accessed similar to an accessor.

When profiling an application using shared memory allocations, be aware that migrations between the host and the device may be occurring implicitly.
These implicit transfers may cause additional apparent latency when launching a kernel (for transfers to the device) or completion latency (for transfers to the host) versus device memory or host memory allocations.

## Command Line Options

| Option | Default Value | Description |
|:--|:-:|:--|
| `-d <index>` | 0 | Specify the index of the SYCL device in the platform to execute on the sample on.
| `-p <index>` | 0 | Specify the index of the SYCL platform to execute the sample on.
