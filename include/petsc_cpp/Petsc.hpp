#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>

#ifdef SLEPC
#include <slepc.h>
#else
#include <petsc.h>
#endif

// forward definitions:
namespace petsc
{
class Vector;
class Matrix;
}
#include <petsc_cpp/Vector.hpp>
#include <petsc_cpp/Matrix.hpp>
#include <petsc_cpp/Utils.hpp>
//#include <petsc_cpp/TimeStepper.hpp>
#ifdef SLEPC
#include <petsc_cpp/EigenvalueSolver.hpp>
#endif


namespace petsc
{

class PetscContext
{
  public:
    PetscContext( const int argc,
                  const char** argv,
                  std::string filename,
                  std::string help )
        : comm_( PETSC_COMM_WORLD )
    {
        int ac = argc;
        char** av = new char* [argc];
        for ( int i = 0; i < argc; i++ ) {
            av[i] = new char[strlen( argv[i] ) + 1];
            std::copy( argv[i], argv[i] + strlen( argv[i] ) + 1, av[i] );
        }
#ifdef SLEPC
        SlepcInitialize( &ac, &av, filename.c_str(), help.c_str() );
#else
        PetscInitialize( &ac, &av, filename.c_str(), help.c_str() );
#endif
    }

    PetscContext( const int argc, const char** argv )
        : comm_( PETSC_COMM_WORLD )
    {
        int ac = argc;
        char** av = new char* [argc];
        for ( int i = 0; i < argc; i++ ) {
            av[i] = new char[strlen( argv[i] ) + 1];
            std::copy( argv[i], argv[i] + strlen( argv[i] ) + 1, av[i] );
        }
#ifdef SLEPC
        SlepcInitialize( &ac, &av, PETSC_NULL, PETSC_NULL );
#else
        PetscInitialize( &ac, &av, PETSC_NULL, PETSC_NULL );
#endif
    }

    int rank() const
    {
        static int rank = [=]() {
            int r;
            MPI_Comm_rank( this->comm_, &r );
            return r;
        }();
        return rank;
    }

    ~PetscContext()
    {
#ifdef SLEPC
        SlepcFinalize();
#else
        PetscFinalize();
#endif
    }

  private:
    MPI_Comm comm_;
};
}
