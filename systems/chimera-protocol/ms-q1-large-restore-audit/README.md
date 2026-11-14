# Full MS Q1 >2GiB restore qualification

The standalone CXX helper loads the full 4,194,304×128 assignment graph through the identical copied owned-device PG_CAGRA implementation. It compares every loaded unrotated dataset float bit to cached `centroids.f32`, in two 32MiB host chunks, then uses that same loaded graph to search the first 65,536 raw token vectors: k=1, itopk=150, batch=65,536. Raw u16→FP32 expansion is copied from the qualified production implementation. All 65,536 labels must equal Q1's first label chunk, with in-range labels and finite distances.

The result binds centroid/graph complete SHA, query FP32 prefix SHA, label/distance SHA, file stats before and after, shape, stride, memory type and degree64. The wrapper binds source, helper, prior verified raw full SHA through unchanged stat, Q1 cache identity/state and library identities. No full raw rehash is performed. The cache and source are read-only.

CPU compile uses hidden CUDA, cores17–18/j2. Five CPU CLI/metadata gates passed; all gates precede PG_CAGRA/CUDA resource construction. Output directories are exclusive. Runtime uses the already validated exact-child exec guard/supervision with inherited output-lock fd and a 180s timeout. It needs a separately authorized offline GPU slot under the existing cohort; this helper does not claim the cohort's global flock.

GPU execution has not started. See `plan.json` for the exact reviewed command and memory model (≤8GiB incremental budget; index payload3GiB plus bounded buffers/search resources).
