#pragma once

#include<memory>
#include<iostream>
#include<array>
#include<mutex>

#ifdef SLEPC
#include<slepc.h>
#else
#include<petsc.h>
#endif

//forward definitions:
namespace petsc{
class Vector;
class Matrix;
}
#include<Vector.hpp>
#include<Matrix.hpp>
#include<Utils.hpp>
#ifdef SLEPC
#include<EigenvalueSolver.hpp>
#endif


namespace petsc {

    class PetscContext {
    public:
        PetscContext( const int argc, const char** argv, std::string filename, std::string help) {
            int ac = argc;
            char** av = new char*[argc];
            for (size_t i = 0; i < argc; i++)
            {
                av[i] = new char[strlen(argv[i])+1];
                std::copy(argv[i], argv[i] + strlen(argv[i])+1, av[i]);
            }
#ifdef SLEPC
            SlepcInitialize(&ac, &av, filename.c_str(), help.c_str());
#else
            PetscInitialize(&ac, &av, filename.c_str(), help.c_str() );
#endif
        }

        PetscContext( const int argc, const char** argv) {
            int ac = argc;
            char** av = new char*[argc];
            for (size_t i = 0; i < argc; i++)
            {
                av[i] = new char[strlen(argv[i])+1];
                std::copy(argv[i], argv[i] + strlen(argv[i])+1, av[i]);
            }
#ifdef SLEPC
            SlepcInitialize(&ac, &av, PETSC_NULL, PETSC_NULL);
#else
            PetscInitialize(&ac, &av, PETSC_NULL, PETSC_NULL );
#endif
        }

        ~PetscContext() {
#ifdef SLEPC
            SlepcFinalize();
#else
            PetscFinalize();
#endif
        }
    };
}
