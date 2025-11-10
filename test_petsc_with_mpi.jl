using MPI
println("Before MPI.Init()")
MPI.Init()
println("After MPI.Init(), before using PETSc")
using PETSc
println("After using PETSc")
