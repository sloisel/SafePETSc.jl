using Test
using MPI
using SafePETSc
SafePETSc.Init()
using PETSc
using SafePETSc.SafeMPI
include(joinpath(@__DIR__, "mpi_test_harness.jl"))
using .MPITestHarness

# PETSc is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nranks = MPI.Comm_size(comm)

# Keep output tidy and aggregate at the end
ts = @testset MPITestHarness.QuietTestSet "Vec_uniform tests" begin

# Test 1: Create a uniform vector
v = ones(16)  # All ranks have the same vector
dr = SafePETSc.Vec_uniform(v)

@test dr isa SafeMPI.DRef
obj = dr.obj
@test obj.v isa PETSc.Vec
@test length(obj.row_partition) == nranks + 1

# Objects are garbage collected automatically via finalizers
# Manually trigger check_and_destroy to ensure collective cleanup
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 2: Check default row partition
partition = SafePETSc.default_row_partition(16, nranks)

@test length(partition) == nranks + 1
@test partition[1] == 1
@test partition[end] == 17

# Check coverage: all rows should be covered exactly once
all_rows = Set()
for i in 0:(nranks-1)
    start = partition[i+1]
    stop = partition[i+2] - 1
    @test start <= stop
    for row in start:stop
        push!(all_rows, row)
    end
end
@test all_rows == Set(1:16)

# Test 3: Custom row partition
v = ones(16)
custom_partition = [1, 5, 9, 13, 17]

if nranks == 4
    dr = SafePETSc.Vec_uniform(v; row_partition=custom_partition)
    @test dr.obj.row_partition == custom_partition
    SafeMPI.check_and_destroy!()
    MPI.Barrier(comm)
end

# Test 4: Verify mpi_uniform assertion works
v_uniform = ones(16)
# This should not error
dr = SafePETSc.Vec_uniform(v_uniform)
@test dr.obj.v isa PETSc.Vec
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 5: Helper constructors
x0 = ones(16)
drx = SafePETSc.Vec_uniform(x0; prefix="x_")

drz = SafePETSc.zeros_like(drx)
z_local = PETSc.unsafe_localarray(drz.obj.v; read=true, write=false)
try
    @test all(z_local[:] .== 0)
finally
    Base.finalize(z_local)
end

dro = SafePETSc.ones_like(drx; prefix="o_")
o_local = PETSc.unsafe_localarray(dro.obj.v; read=true, write=false)
try
    @test all(o_local[:] .== 1)
finally
    Base.finalize(o_local)
end
@test dro.obj.prefix == "o_"

drf = SafePETSc.fill_like(drx, 7.0)
f_local = PETSc.unsafe_localarray(drf.obj.v; read=true, write=false)
try
    @test all(f_local[:] .== 7.0)
finally
    Base.finalize(f_local)
end

# cleanup - objects are garbage collected automatically
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 6: + and - sugar
x0 = ones(16)
y0 = fill(2.0, 16)
drx = SafePETSc.Vec_uniform(x0)
dry = SafePETSc.Vec_uniform(y0)

drz = drx + dry
z_local = PETSc.unsafe_localarray(drz.obj.v; read=true, write=false)
try
    @test all(z_local[:] .== 3.0)
finally
    Base.finalize(z_local)
end

drw = dry - drx
w_local = PETSc.unsafe_localarray(drw.obj.v; read=true, write=false)
try
    @test all(w_local[:] .== 1.0)
finally
    Base.finalize(w_local)
end

drn = -drx
n_local = PETSc.unsafe_localarray(drn.obj.v; read=true, write=false)
try
    @test all(n_local[:] .== -1.0)
finally
    Base.finalize(n_local)
end

drp = 5 .+ drx  # scalar + Vec via broadcast (works)
p_local = PETSc.unsafe_localarray(drp.obj.v; read=true, write=false)
try
    @test all(p_local[:] .== 6.0)
finally
    Base.finalize(p_local)
end

# cleanup - objects are garbage collected automatically
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

# Test 7: In-place broadcasting
x0 = ones(16)
y0 = zeros(16)
drx = SafePETSc.Vec_uniform(x0; prefix="x_")
dry = SafePETSc.Vec_uniform(y0; prefix="y_")

# y .= 2 .* x .+ 3 (DRef-aware broadcast)
dry .= 2 .* drx .+ 3

# Check local result
y_local = PETSc.unsafe_localarray(dry.obj.v; read=true, write=false)
try
    @test all(y_local[:] .== 5)
finally
    Base.finalize(y_local)
end

# y .= y .+ x
dry .= dry .+ drx
y_local = PETSc.unsafe_localarray(dry.obj.v; read=true, write=false)
try
    @test all(y_local[:] .== 6)
finally
    Base.finalize(y_local)
end

# Out-of-place: z = x .+ 4
drz = drx .+ 4
@test drz isa SafeMPI.DRef
z_local = PETSc.unsafe_localarray(drz.obj.v; read=true, write=false)
try
    @test all(z_local[:] .== 5)
finally
    Base.finalize(z_local)
end

# cleanup - objects are garbage collected automatically
SafeMPI.check_and_destroy!()
MPI.Barrier(comm)

end  # End of QuietTestSet

# Aggregate per-rank counts and print a single summary on root
local_counts = [
    get(ts.counts, :pass, 0),
    get(ts.counts, :fail, 0),
    get(ts.counts, :error, 0),
    get(ts.counts, :broken, 0),
    get(ts.counts, :skip, 0),
]

global_counts = similar(local_counts)
MPI.Reduce!(local_counts, global_counts, +, 0, comm)

if rank == 0
    println("Test Summary: Vec_uniform tests (aggregated across $(nranks) ranks)")
    println("  Pass: $(global_counts[1])  Fail: $(global_counts[2])  Error: $(global_counts[3])  Broken: $(global_counts[4])  Skip: $(global_counts[5])")
end

MPI.Barrier(comm)

if global_counts[2] > 0 || global_counts[3] > 0
    Base.exit(1)
end

# Ensure all ranks reach this point before deciding outcome
MPI.Barrier(comm)


# Note: We don't call MPI.Finalize() here because Julia's MPI.jl
# automatically finalizes MPI at exit via atexit hook
