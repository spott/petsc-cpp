#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <cassert>


namespace petsc
{

// enum class ProblemType { Hermitian, GeneralizedHermitian, NonHermitian,
// GeneralizedNonHermitian, PositiveGeneralizedNonHermitian,
// GeneralizedHermitianIndefinite };
// template<enum PT>
// We are only going to look at non-generalized solvers initially, later we
// will
// template this with the above enum.
class EigenvalueSolver
{
  public:
    // EigenvalueSolver(const MPI_Comm& comm, Matrix& A, Matrix& B, int
    // dim,
    // EPSProblemType type, EPSWhich which_pair) {
    // EPSCreate(comm, &e_);
    // EPSSetOperators(e_, A.m_, B.m_);
    // EPSSetProblemType(e_, type);
    // EPSSetDimensions(e_, dim, PETSC_DECIDE , PETSC_DECIDE);
    // EPSSetWhichEigenpairs(e_, which_pair);
    // EPSSetFromOptions(e_);
    // EPSSolve(e_);
    //}

    EigenvalueSolver( const MPI_Comm& comm,
                      Matrix& A,
                      int dim,
                      EPSWhich which_pair = EPS_SMALLEST_REAL,
                      EPSProblemType type = EPS_HEP )
        : op_( A )
    {
        assert( type == EPS_HEP ||
                type == EPS_NHEP ); // we only want hermitian
                                    // or non-hermitian
                                    // values
        EPSCreate( comm, &e_ );
        EPSSetOperators( e_, A.m_, PETSC_NULL );
        EPSSetProblemType( e_, type );
        EPSSetDimensions( e_, dim, PETSC_DECIDE, PETSC_DECIDE );
        EPSSetWhichEigenpairs( e_, which_pair );
        EPSSetFromOptions( e_ );
        EPSSolve( e_ );
    }
    EigenvalueSolver( Matrix& A, int dim = 1 )
        : EigenvalueSolver( A.comm(), A, dim, EPS_SMALLEST_REAL, EPS_HEP )
    {
    }

    // rule of 4.5...
    EigenvalueSolver( const EigenvalueSolver& other ) =
        delete; // we don't need no stinkin copy constructor...

    EigenvalueSolver( EigenvalueSolver&& other )
        : e_( other.e_ ), op_( other.op_ )
    {
        other.e_ = PETSC_NULL;
        // other.op_ = nullptr;
    }

    // EigenvalueSolver& operator=( EigenvalueSolver&& other )
    // {
    //     swap( *this, other );
    //     return *this;
    // }

    friend void swap( EigenvalueSolver& first,
                      EigenvalueSolver& second ); // nothrow

    ~EigenvalueSolver() { EPSDestroy( &e_ ); }

    // Getters:
    MPI_Comm comm() const;

  Matrix op() const;
    // can't controll the order of the destruction of static objects...
    // so this object can't be static, because it won't be destroyed till
    // it is
    // too late.

    int iteration_number() const;

    std::array<int, 3> dimensions() const;

    std::tuple<PetscReal, PetscInt> tolerances() const;

    int num_converged() const;

    struct result {
        int nev;
        PetscScalar evalue;
        Vector evector;
    };

    // print!:
    void print() const;

    // This version spits out a new vector:
    EigenvalueSolver::result get_eigenpair( int nev );

    class Iterator : public std::iterator<std::forward_iterator_tag,
                                          EigenvalueSolver::result,
                                          int>
    {
      public:
        int nev;
        EigenvalueSolver& e;

        explicit Iterator( const int i, EigenvalueSolver& es )
            : nev( i ), e( es )
        {
        }

        Iterator( const Iterator& i ) : nev( i.nev ), e( i.e ){};

        Iterator( Iterator&& i ) : nev( i.nev ), e( i.e ) {}

        Iterator& operator=( Iterator rhs );

        Iterator& operator++();
        Iterator operator++( int );

        // iterator& operator+=(iterator::Distance diff) {
        // nev+=diff;
        // return *this;
        //}
        // iterator& operator--() {
        // nev--;
        // return *this;
        //}
        // iterator& operator--(int) {
        // auto ret = *this;
        // nev--;
        // return ret;
        //}

        // iterator& operator-=(iterator::Distance diff) {
        // nev-=diff;
        // return *this;
        //}


        bool operator!=( const Iterator& rhs );

        Iterator::value_type operator*();
    };

    EigenvalueSolver::Iterator begin();

    EigenvalueSolver::Iterator end();

    EPS e_;
    Matrix& op_;
};
}
