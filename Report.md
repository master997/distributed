# 6G6Z0038 Distributed Computing — 1CWK100 Report
**Author:** [your name + ID]  •  **Submitted:** 5 May 2026

---

## Section 1 — Concepts of distributed computing applied (LO1)

The pipeline performs three operations in sequence on an N×N matrix: **transposition**, **3×3 zone sum**, and **self-multiplication**. Each operation was first implemented sequentially to verify correctness against the supplied example matrices, then parallelised. The following module concepts were applied directly:

**`std::thread` and `.join()` (DC-2 lecture).**  Each operation is parallelised by partitioning its outer row-loop into `T = std::thread::hardware_concurrency()` row-strips. Each strip is an independent unit of work — every worker writes to a disjoint set of destination rows, so there is no shared-write hazard and no `std::mutex` is required (the safety reasoning comes from DC-3 lecture: disjoint memory writes are race-free by definition). The row-strip pattern is exactly the construct from DC-2 Tutorial Example 12a (vector of threads filled in a `for` loop, joined in a second `for` loop), generalised from one-thread-per-row to one-thread-per-row-strip so the work granularity matches the number of cores.

**Thread pool (DC-8 lecture).**  After confirming correctness with bare `std::thread`, the implementation moved to the `ThreadPool` class shown in DC-8 lecture slides 6–10. Each call to an operation now submits T tasks via `pool.enqueue(...)` and waits on their `std::future` results, instead of spawning fresh threads per call. The pool itself is constructed lazily on first use as a function-static instance (`globalPool()`) and reused across all 10 timed iterations and all three operations within each iteration. This eliminates roughly 250 `std::thread` constructions per program run (≈ 10 calls × 3 operations × 8 strips), saving the per-thread spawn cost of 30–100 µs that would otherwise sit in the timed path.

**`std::condition_variable` and `std::mutex` (DC-3 + DC-6 lectures).**  Although the matrix operations themselves do not need a mutex, the thread-pool internals do: workers sleep on a condition variable bound to the queue mutex (`condition.wait(lock, predicate)`), are notified by `condition.notify_one()` whenever a new task is enqueued, and the dispatcher uses RAII `std::unique_lock` to manage critical sections (DC-3 lecture pattern). This is the producer/consumer paradigm from DC-7 lecture, with the dispatcher as producer and the workers as consumers.

**`std::future` (DC-5 lecture).**  Each `pool.enqueue` call returns a future. Operations collect their futures into a `std::vector<std::future<void>>` and call `.get()` on each in turn — a clean per-task join that also propagates any exception that escaped a worker. This replaces the manual `for (auto& th : threads) th.join()` pattern with a more idiomatic, future-based equivalent.

**Profiling with `std::chrono` (DC-4 lecture).**  Per-operation timings are accumulated into static `double` counters across the 10 timed runs and printed once at the end of the program, on `stderr` so they don't pollute the assessor's `Average:` line. The measured breakdown at N=1000 is **op1 transpose ≈ 1.1 ms**, **op2 zone_sum ≈ 0.7 ms**, **op3 matmul ≈ 55 ms** — matmul accounts for ~96% of total time, so further optimisation effort is focused there.

**Speed-up estimate (DC-4 lecture, Amdahl's law).**  Compared to the shipped baseline (sequential dummy operations dominated by file-write I/O at ≈ 117 ms for N=300), the final implementation runs in ≈ 0.4 ms at the same N — a **>290× improvement**, the bulk of which comes from removing in-loop file I/O (the spec's "remove debug printing" rule) and the parallel matmul. At N=1000 the matmul-only speed-up from sequential to parallel is **3.2×** on 8 cores, against a theoretical Amdahl ceiling of 8× (limited by memory-bandwidth contention and the sequential portions outside the operation).

---

## Section 2 — Trade-offs (LO3)

| Trade-off | Option A | Option B | Measured @ N=1000 | Choice + reason |
|-----------|----------|----------|-------------------|-----------------|
| **In-loop file I/O** | Keep `fileWrite` calls inside `matrixOperationsInit` (shipped) | Hide them behind `-DVERIFY` so they only run during correctness checks | A: ≈ 117 ms (N=300), B: ≈ 0.3 ms (N=300) | **B.** Disk I/O is roughly 4-5 orders of magnitude slower than memory access. With I/O in the loop, the assessor would have been measuring SSD throughput, not algorithm performance. The spec is explicit ("remove any debug printing"), but the magnitude of the win was the larger reason. |
| **Matmul loop order** | Textbook `(i, j, k)` | Cache-aware `(i, k, j)` | Internal A/B at small N showed (i,j,k) ~3-5× slower (column-stride access). | **B.** Both orders compute the same result, but `(i, k, j)` makes the inner `j`-loop walk one row of `dst` and one row of `src` in step. Both are sequential row-major accesses, so each cache line is reused for many iterations before being evicted. The textbook order strides through src by N×8 bytes per `k`, defeating the prefetcher. |
| **Threading granularity** | One task per row | One task per row-strip (N/T contiguous rows) | A: heavy task-dispatch overhead (~N enqueue calls per op), B: only T enqueue calls per op | **B.** Row-strips because T (≈ 8) is much smaller than N (1000), so we make 8 enqueue calls instead of 1000. Strips also share L2 cache lines between rows, whereas one-row-per-task could end up scheduled across distant cores. |
| **Thread spawn model** | Fresh `std::thread` per operation call (Change 6–8) | Static thread pool reused across all calls (Change 10) | A: 58.1 ms (3 ops × 8 spawns × 10 iterations = 240 spawns), B: 57.2 ms | **B.** The win at N=1000 is small (~1.5%) because matmul itself is long enough to amortise the 30–100 µs spawn cost. The win is much larger at smaller N or shorter operations (zone_sum dropped from 0.91 ms → 0.73 ms, ~20%). The pool also gives clean futures-based synchronisation and is the rubric-named "advanced concept". |
| **Intermediate buffer allocation** | Construct three N×N `vector<vector<double>>` per call (shipped) | Static buffers, allocated once on the first call and reused | A: 379 ms, B: 182 ms | **B.** At N=1000 each intermediate is ~8 MB. The shipped version allocated ~24 MB three times per call × 10 calls = ~720 MB of allocator traffic per program. Static buffers move that allocation out of the timed path entirely — a –52% win. |
| **Sequential fast-path for tiny N** | Always go through the thread pool | If N < 64, run inline | A: ≈ 0.06 ms at N=10 (all dispatch overhead), B: ≈ 0.002 ms at N=10 | **B.** Below ~64 the cost of submitting tasks to the pool exceeds the arithmetic. The fast-path doesn't help at N=1000 (we never hit it), but protects performance if the assessor tests at small sizes. |

---

## What was deliberately NOT implemented (and why)

- **SIMD intrinsics, OpenMP, Apple Accelerate, Eigen, BLAS** — disallowed by the spec ("Libraries that provide pre-developed parallelised operations … are not permitted").
- **Cache-blocked / tiled matmul** — would have provided further speed-up at large N, but was not stable enough within the hackathon time budget. Documented here as an opportunity for future work.
- **Merged transpose + zone_sum** — possible to compute `op2[i][j] = Σ src[j±1][i±1]` directly, eliminating the materialised `op1` matrix. Same idea as the rowSum + colSum trick from DC-11. Tried briefly, but the index arithmetic was error-prone within the time budget; reverted to the cleaner two-pass version.
