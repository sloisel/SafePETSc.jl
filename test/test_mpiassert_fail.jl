using MPI
using SafePETSc
SafePETSc.Init()
using SafePETSc.SafeMPI

# MPI is initialized by SafePETSc.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# One failing rank (rank==2): cond==false -> assert fails
ok_some = (rank != 2)
@mpiassert ok_some "rank 2 intentionally failing"

# Should never reach here due to abort
println("[rank $rank] unexpected: test did not abort")

# Finalize SafeMPI to prevent shutdown race conditions
SafeMPI.finalize()
