// device_record_kernel.cu
//
// Kernels behind the opguard::probe operator (opguard/device_record).
//
// * dr_xor_kernel: regional flat-word XOR fingerprint of one tensor, accumulated
//   with atomicXor into that tensor's 4 scratch words. It matches
//   opguard/tracer/xor_signature.py (_flat_word_signature, regions=4): bytes are
//   packed little-endian into uint32 words, the final partial word is
//   zero-padded, words are split into 4 contiguous regions with the remainder
//   going to the earlier regions. XOR is order independent, so the result does
//   not depend on the launch grid.
// * dr_commit_kernel: launched exactly once per probe call. Claims a global
//   sequence number s and a per-site number k, writes one slot of the record
//   ring (whose address/capacity it reads from device memory, never from a
//   captured kernel argument), moves the fingerprints out of scratch (leaving it
//   zero) and publishes the slot by writing seq last.
//
// This file intentionally has no device globals or static initialisation; it
// allocates nothing and does nothing until a launcher is called.

#include <cuda_runtime.h>
#include <stdint.h>

namespace {

constexpr int kXorThreads = 256;
constexpr int kXorWarps = kXorThreads / 32;
constexpr int kCommitThreads = 128;
constexpr int64_t kMaxXorBlocks = 1024;

// Slot layout, in int64 words: seq, site, n, k, call_id, replay_epoch, then
// T_max * 4 uint32 fingerprint words (two per int64, little endian).
constexpr int kSlotSeq = 0, kSlotSite = 1, kSlotN = 2, kSlotK = 3, kSlotCallId = 4,
              kSlotEpoch = 5, kSlotHeader = 6;

__device__ __forceinline__ uint32_t dr_warp_xor(uint32_t v) {
#pragma unroll
    for (int ofs = 16; ofs > 0; ofs >>= 1) v ^= __shfl_down_sync(0xFFFFFFFFu, v, ofs);
    return v;
}

template <bool Aligned>
__device__ __forceinline__ uint32_t dr_load_word(const uint8_t* bytes, uint64_t nbytes, uint64_t word) {
    const uint64_t b = word * 4;
    if (Aligned && b + 4 <= nbytes) return reinterpret_cast<const uint32_t*>(bytes)[word];
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i)
        if (b + i < nbytes) v |= static_cast<uint32_t>(bytes[b + i]) << (8 * i);
    return v;
}

// b1, b2, b3: first word index of regions 1, 2 and 3.
template <bool Aligned>
__global__ void dr_xor_kernel(const uint8_t* __restrict__ bytes, uint64_t nbytes,
                              uint64_t b1, uint64_t b2, uint64_t b3,
                              uint32_t* __restrict__ out4,
                              const int32_t* __restrict__ enable, int64_t site) {
    if (enable[site] == 0) return;
    const uint64_t words = (nbytes + 3) / 4;
    uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const uint64_t stride = static_cast<uint64_t>(gridDim.x) * blockDim.x;
    for (uint64_t w = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x; w < words; w += stride) {
        const uint32_t v = dr_load_word<Aligned>(bytes, nbytes, w);
        if (w < b1) a0 ^= v;
        else if (w < b2) a1 ^= v;
        else if (w < b3) a2 ^= v;
        else a3 ^= v;
    }
    a0 = dr_warp_xor(a0); a1 = dr_warp_xor(a1); a2 = dr_warp_xor(a2); a3 = dr_warp_xor(a3);

    __shared__ uint32_t partial[4][kXorWarps];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) {
        partial[0][warp] = a0; partial[1][warp] = a1; partial[2][warp] = a2; partial[3][warp] = a3;
    }
    __syncthreads();
    if (warp == 0) {
        // Lanes 0..3 each finish one region.
        if (lane < 4) {
            uint32_t acc = 0;
#pragma unroll
            for (int i = 0; i < kXorWarps; ++i) acc ^= partial[lane][i];
            if (acc) atomicXor(out4 + lane, acc);
        }
    }
}

__global__ void dr_commit_kernel(int64_t site, int64_t n, int64_t t_site, int64_t call_id,
                                 int64_t slot_words,
                                 const int64_t* __restrict__ ring_desc,
                                 unsigned long long* __restrict__ cursor,
                                 unsigned long long* __restrict__ site_counters,
                                 uint32_t* __restrict__ scratch,  // already offset to this site
                                 const int32_t* __restrict__ enable,
                                 const int64_t* __restrict__ replay_epoch) {
    const int tid = threadIdx.x;
    if (enable[site] == 0) {
        // Keep "scratch is zero after every probe" independent of when the
        // switch flipped relative to the XOR launches. No number is claimed.
        for (int64_t i = tid; i < t_site * 4; i += blockDim.x) scratch[i] = 0u;
        return;
    }

    __shared__ int64_t* sh_slot;
    __shared__ unsigned long long sh_s;
    if (tid == 0) {
        int64_t* base = reinterpret_cast<int64_t*>(ring_desc[0]);
        const int64_t cap = ring_desc[1];
        const unsigned long long s = atomicAdd(cursor, 1ull);
        const unsigned long long k = atomicAdd(site_counters + site, 1ull);
        int64_t* slot = nullptr;
        if (cap > 0 && base != nullptr) {
            slot = base + static_cast<int64_t>(s % static_cast<unsigned long long>(cap)) * slot_words;
            slot[kSlotSite] = site;
            slot[kSlotN] = n;
            slot[kSlotK] = static_cast<int64_t>(k);
            slot[kSlotCallId] = call_id;
            slot[kSlotEpoch] = *replay_epoch;
        }
        sh_slot = slot;
        sh_s = s;
    }
    __syncthreads();
    int64_t* slot = sh_slot;
    const int64_t nwords = n * 4;
    if (slot == nullptr) {
        // No ring (capacity 0): write nothing; the host reports this call as lost.
        for (int64_t i = tid; i < nwords; i += blockDim.x) scratch[i] = 0u;
        return;
    }
    uint32_t* fp = reinterpret_cast<uint32_t*>(slot + kSlotHeader);
    for (int64_t i = tid; i < nwords; i += blockDim.x) {
        fp[i] = scratch[i];
        scratch[i] = 0u;
    }
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        __threadfence();
        *reinterpret_cast<volatile long long*>(slot + kSlotSeq) = static_cast<long long>(sh_s);
    }
}

}  // namespace

extern "C" void dr_launch_xor(const uint8_t* bytes, uint64_t nbytes, uint32_t* out4,
                              const int32_t* enable, int64_t site, cudaStream_t stream) {
    if (nbytes == 0) return;  // scratch is already zero, which is the empty fingerprint
    const uint64_t words = (nbytes + 3) / 4;
    const uint64_t base = words / 4, rem = words % 4;
    const uint64_t b1 = base + (rem > 0);
    const uint64_t b2 = b1 + base + (rem > 1);
    const uint64_t b3 = b2 + base + (rem > 2);
    int64_t blocks = static_cast<int64_t>((words + kXorThreads - 1) / kXorThreads);
    if (blocks > kMaxXorBlocks) blocks = kMaxXorBlocks;
    if (blocks < 1) blocks = 1;
    if ((reinterpret_cast<uintptr_t>(bytes) & 3u) == 0)
        dr_xor_kernel<true><<<static_cast<unsigned>(blocks), kXorThreads, 0, stream>>>(
            bytes, nbytes, b1, b2, b3, out4, enable, site);
    else
        dr_xor_kernel<false><<<static_cast<unsigned>(blocks), kXorThreads, 0, stream>>>(
            bytes, nbytes, b1, b2, b3, out4, enable, site);
}

extern "C" void dr_launch_commit(int64_t site, int64_t n, int64_t t_site, int64_t call_id,
                                 int64_t slot_words, const int64_t* ring_desc, int64_t* cursor,
                                 int64_t* site_counters, uint32_t* site_scratch,
                                 const int32_t* enable, const int64_t* replay_epoch,
                                 cudaStream_t stream) {
    dr_commit_kernel<<<1, kCommitThreads, 0, stream>>>(
        site, n, t_site, call_id, slot_words, ring_desc,
        reinterpret_cast<unsigned long long*>(cursor),
        reinterpret_cast<unsigned long long*>(site_counters),
        site_scratch, enable, replay_epoch);
}
