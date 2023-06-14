/*
// Copyright (c) 2023 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl/sycl.hpp>
#include <popl/popl.hpp>

#include <stdio.h>

struct Node {
    Node() :
        next( nullptr ),
        value( 0xDEADBEEF ) {}

    Node*       next;
    uint32_t    value;
};

int main(int argc, char** argv)
{
    int deviceIndex = 0;
    size_t numNodes = 4;

    {
        popl::OptionParser op("Supported Options");
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<size_t>>("n", "nodes", "Number of Linked List Nodes", numNodes, &numNodes);        
        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: smemlinkedlist [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    auto devices = sycl::device::get_devices();
    if (deviceIndex >= devices.size()) {
        fprintf(stderr, "Error: device index %d is unavailable, only %zu devices found.\n",
            deviceIndex, devices.size());
        return -1;
    }

    auto device = devices[deviceIndex];
    auto platform = device.get_platform();

    printf("Running on SYCL platform: %s\n", platform.get_info<sycl::info::platform::name>().c_str());
    printf("Running on SYCL device: %s\n", device.get_info<sycl::info::device::name>().c_str());

    printf("Initializing tests...\n");

    auto queue = sycl::queue{ device, sycl::property::queue::in_order() };

    printf("Building the linked list...\n");
    Node*   h_head = nullptr;
    {
        Node* current = nullptr;
        for (size_t i = 0; i < numNodes; i++) {
            if (i == 0) {
                h_head = sycl::malloc_shared<Node>(1, queue);
                current = h_head;
            }
            
            if (current != nullptr) {
                current->value = i * 2;
                if (i != numNodes - 1) {
                    current->next = sycl::malloc_shared<Node>(1, queue);
                } else {
                    current->next = nullptr;
                }

                current = current->next;
            }
        }
    }

    printf("Updating the linked list on the device...\n");
    {
        queue.single_task([=]() {
            Node* current = h_head;
            while (current != nullptr) {
                current->value *= 2;
                current = current->next;
            }
        });
    }

    printf("Verifying results...\n");
    {
        queue.wait();

        const Node* current = h_head;
        size_t mismatches = 0;
        for (size_t i = 0; i < numNodes; i++) {
            if (current == nullptr) {
                mismatches++;
            } else {
                if (current->value != i * 4) {
                    mismatches++;
                }
                current = current->next;
            }
        }

        if (mismatches) {
            fprintf(stderr, "Error: Found %zu mismatches out of %zu values!\n",
                mismatches, numNodes);
        } else {
            printf("Success.\n");
        }
    }

    // TODO: Should really clean up and free nodes here...

    printf("... done!\n");
    return 0;
}
