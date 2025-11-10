using MPI

  # This is the problematic sequence:
  # 1. Load PETSc BEFORE calling MPI.Init()
  using PETSc  # <-- PETSc.__init__() runs here, tries to access MPI

  # 2. Initialize MPI (too late!)
  if !MPI.Initialized()
      MPI.Init()
  end

  # 3. Initialize PETSc
  if !PETSc.initialized(PETSc.petsclibs[1])
      PETSc.initialize()
  end

  println("If we got here, it worked!")
