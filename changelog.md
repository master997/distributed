# Hackathon changelog — 5 May 2026

Each row is one change. Test point is N=1000 unless noted. "Δ vs prev" is the average-time change relative to the previous row.

| # | Change | N | Average ms | Δ vs prev | Correct? | Notes |
|---|--------|---|------------|-----------|----------|-------|
| 0 | Baseline as shipped (3 dummy copy-ops, 5 file writes inside timed path) | 300 | 117.305 | — | N/A | All time is file I/O. Real algorithm cost is hidden. |
| 1 | operation1 = transpose (sequential, [] access) | 300 | 63.554 | -45.8% | ✓ transpose | op1 verified vs ExampleMatrices1. Faster than baseline because op2/op3 still write the smaller transposed result instead of the original src — incidental, not a real speedup. |

## Per-operation breakdown (after Change 9)

_(populated once Change 9 lands)_

## Failed experiments (these become trade-off rows in the report)

_(none yet)_
