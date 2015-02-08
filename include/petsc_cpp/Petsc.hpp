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
    {
        int ac = argc;
        char** av = new char* [argc + 1];
        for ( int i = 0; i < argc; i++ ) {
            av[i] = new char[strlen( argv[i] ) + 1];
            std::copy( argv[i], argv[i] + strlen( argv[i] ) + 1, av[i] );
        }
        av[ac] = NULL;
#ifdef SLEPC
        SlepcInitialize( &ac, &av, filename.c_str(), help.c_str() );
#else
        PetscInitialize( &ac, &av, filename.c_str(), help.c_str() );
#endif
        comm_ = PETSC_COMM_WORLD;
    }

    PetscContext( const int argc, const char** argv )
    {
        int ac = argc;
        char** av = new char* [argc + 1];
        for ( int i = 0; i < argc; i++ ) {
            av[i] = new char[strlen( argv[i] ) + 1];
            std::copy( argv[i], argv[i] + strlen( argv[i] ) + 1, av[i] );
        }
        av[ac] = NULL;
#ifdef SLEPC
        SlepcInitialize( &ac, &av, PETSC_NULL, PETSC_NULL );
#else
        PetscInitialize( &ac, &av, PETSC_NULL, PETSC_NULL );
#endif
        comm_ = PETSC_COMM_WORLD;
    }

    MPI_Comm comm() const { return comm_; }

    int rank() const
    {
        int r;
        MPI_Comm_rank( this->comm_, &r );
        return r;
    }

    ~PetscContext()
    {
#ifdef SLEPC
        SlepcFinalize();
#else
        PetscFinalize();
#endif
    }

    // private:
    MPI_Comm comm_;
};
}
