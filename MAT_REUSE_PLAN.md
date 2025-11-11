Here is the plan. Please work on this plan and update this file to mark items that are done, as you progress.

i would like to allow spdiagm to be able to draw matrices from the non-product pool (in src/mat.jl). right now it cannot. to that end it should work as follows.
 1) DONE: for the sparsity pattern, spdiagm now always allocates the full diagonals as structural nonzeros, inserting explicit zeros where needed so the entire diagonal is present in the structure.
 2) DONE: implemented _structure_fingerprint via a new _local_structure_hash helper. _matrix_fingerprint is now a thin wrapper over _structure_fingerprint.
 3) DONE: added lookup against the non-product pool by structure fingerprint, prefix, and partitions. Pooled entries now carry their own fingerprint for O(1) matching.
 4) DONE: if a suitable Mat is found, spdiagm reuses it and assembles directly into it; otherwise it falls back to Mat_sum.
 5) DONE: added Base.:+ and Base.:- for Mat, which compute the union sparsity, look up a compatible non-product pooled matrix by fingerprint, and reuse it when available (using MatAXPY under the hood).
 6) DONE: ran Pkg.test(); all tests, including spdiagm, passed on 4 ranks.
