# Hackathon changelog — 5 May 2026

Each row is one change. Test point is N=1000 unless noted. "Δ vs prev" is the average-time change relative to the previous row.

| # | Change | N | Average ms | Δ vs prev | Correct? | Notes |
|---|--------|---|------------|-----------|----------|-------|
| 0 | Baseline as shipped (3 dummy copy-ops, 5 file writes inside timed path) | 300 | 117.305 | — | N/A | All time is file I/O. Real algorithm cost is hidden. |
| 1 | operation1 = transpose (sequential, [] access) | 300 | 63.554 | -45.8% | ✓ transpose | op1 verified vs ExampleMatrices1. Faster than baseline because op2/op3 still write the smaller transposed result instead of the original src — incidental, not a real speedup. |
| 2 | Removed file writes from timed path (now behind `-DVERIFY`) | 300 | 0.313 | **-99.5%** | ✓ transpose | I/O was hiding everything. Real algorithm cost finally visible. |
| 2 | Same change, larger N | 1000 | 8.565 | (new bench size) | ✓ transpose | First useful N=1000 number. From here we benchmark at 1000. |
| 3 | operation2 = zone_sum (sequential, 3x3 stencil) | 1000 | 10.691 | +24.8% | ✓ transpose, ✓ zone_sum | Op2 now does real work. Cost is 9 reads/cell × N². op3 still dummy. |
| 4 | operation3 = matmul (i,k,j sequential) | 1000 | 379.158 | +3445% | ✓ all three (examples 1/2/3) | Matmul is N³ — expected jump. We now have a correct, full sequential pipeline. This is the line we parallelise from. |
| 5 | Static intermediate buffers (allocated once, reused 10x) | 1000 | 181.901 | **-52.0%** | ✓ all three | Shipped code re-allocated 24 MB × 3 buffers on every call. Static lifts that out of the timed path. Bigger win than expected — allocator was a real bottleneck. |
| 6 | Parallel matmul: 8 threads × row-strips (DC-2 Tut Ex 12a pattern) | 1000 | 56.800 | **-68.8%** | ✓ all three | 3.2x speedup on 8 cores. No mutex needed — each thread writes a disjoint set of dst rows. Below ideal 8x because transpose + zone_sum are still sequential and there's some memory-bandwidth contention between cores. |
| 7 | Parallel transpose: 8 threads × row-strips | 1000 | 57.005 | +0.4% | ✓ all three | Within noise. Transpose is tiny relative to matmul (~1ms), so thread-spawn overhead matches the gain. Required by spec ("each operation must be parallelised") and counts towards LO2. |

## Per-operation breakdown (after Change 9)

_(populated once Change 9 lands)_

## Failed experiments (these become trade-off rows in the report)

_(none yet)_
