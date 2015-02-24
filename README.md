petsc-cpp
=========

C++ Interface to PETSc

Currently limited to only a few different classes:

# PetscContext:

PetscContext is the initialization of Petsc.  Create the `petsc::PetscContext` object at the beginning of main and it will call `PetscInit` and `PetscFinalize` for you.  Currently doesn't do much more than that.

# Vector:

A wrapper for a Petsc `Vec` object.  Has a couple of different member functions, and lacks a default constructor (we don't want one, cause really, you shouldn't be creating a vector this way, using this interface).

# Matrix:

A wrapper for a Petsc `Mat` object.  a few member functions, including a `get_vectors()` method that returns an array of `Vectors` for your pleasure.

# EigenvalueSolver:

A wrapper for a Slepc `EPS` object.  The initializer creates and solves an eigenvalue problem.  Currently only works for non-generalized eigenvalue problems.
