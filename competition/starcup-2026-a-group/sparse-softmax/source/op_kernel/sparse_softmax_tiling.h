#pragma once
#include <cstdint>

struct SparseSoftmaxTilingData {
    // Only the fields the kernel actually reads are kept (lean 32-byte ABI).
    // The device reads the whole struct once per block with in-order scalar loads,
    // so halving the struct halves that per-block entry cost. The struct stays
    // 32 B and 8-byte aligned.
    uint32_t tasks;
    uint32_t axisLength;
    uint32_t innerSize;
    uint32_t mode;           // 1=index, 2=CSR ptr
    uint32_t ptrLength;
    uint32_t indexBroadcast;
    float    eps;
    uint32_t pad0;           // id marker for the 64-bit routes (1=index, 2=ptr, 0=ia32)
};
static_assert(sizeof(SparseSoftmaxTilingData) == 32, "lean tiling ABI");

// axisLength > CAP (4096): the streaming kernel above keeps its per-group state
// in the user workspace, one slot per block. A slot holds three float tables
// (group max | group sum | Kahan compensation) of LARGE_AXIS_TABLE_GROUPS
// entries each; the host reserves LARGE_AXIS_SLOT_BYTES per block through
// GetWorkspaceSizes. No group state is needed when the id span (index mode) or
// the CSR interval count (ptr mode) exceeds the table capacity - the kernel
// then falls back to the element-exact scan.
constexpr uint32_t LARGE_AXIS_TABLE_GROUPS = 32768u;
constexpr uint32_t LARGE_AXIS_SLOT_BYTES = 3u * LARGE_AXIS_TABLE_GROUPS * sizeof(float);
