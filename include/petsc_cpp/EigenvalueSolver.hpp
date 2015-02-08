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

    EigenvalueSolver( Matrix& A,
                      Matrix* space,
                      int dim,
                      EPSWhich which_pair = EPS_SMALLEST_REAL,
                      EPSProblemType type = EPS_HEP )
        : op_( A ), inner_product_space( space )
    {
        assert( type == EPS_HEP ||
                type == EPS_NHEP ); // we only want hermitian
                                    // or non-hermitian
                                    // values
        EPSCreate( A.comm(), &e_ );
        EPSSetOperators( e_, A.m_, PETSC_NULL );
        EPSSetProblemType( e_, type );
        EPSSetDimensions( e_, dim, PETSC_DECIDE, PETSC_DECIDE );
        EPSSetWhichEigenpairs( e_, which_pair );
        EPSSetFromOptions( e_ );
    }
    EigenvalueSolver( Matrix& A, int dim = 1 )
        : EigenvalueSolver( A, nullptr, dim, EPS_SMALLEST_REAL, EPS_HEP )
    {
    }

    // rule of 4.5...
    EigenvalueSolver( const EigenvalueSolver& other ) =
        delete; // we don't need no stinkin copy constructor...

    EigenvalueSolver( EigenvalueSolver&& other )
        : e_( other.e_ ), op_( other.op_ ),
          inner_product_space( other.inner_product_space )
    {
        other.e_ = PETSC_NULL;
        other.inner_product_space = nullptr;
        // other.op_ = nullptr;
    }

    friend void swap( EigenvalueSolver& first,
                      EigenvalueSolver& second ); // nothrow

    ~EigenvalueSolver() { EPSDestroy( &e_ ); }

    // Getters:
    MPI_Comm comm() const;

    Matrix op() const;

    void set_inner_product_space( Matrix B );
    void solve();
    void set_initial_vector( Vector init );
    void
    balance( EPSBalance bal, int iter, double cuttoff = PETSC_DECIDE );
    void shift_invert( std::complex<double> sigma );

    void dimensions( int nev, int mpd = -1, int ncv = -1 );

    int iteration_number() const;

    std::array<int, 3> dimensions() const;

    std::tuple<PetscReal, PetscInt> tolerances() const;
    void tolerances( double tol, int iterations );

    int num_converged() const;

    struct result {
        int nev;
        PetscScalar evalue;
        Vector evector;
    };

    // print!:
    void print() const;
    void save_basis( const std::string& filename ) const;
    void save_basis( const std::string& filename,
                     std::function<void(Vector&)> modification ) const;

    // This version spits out a new vector:
    EigenvalueSolver::result get_eigenpair( int nev ) const;
    PetscScalar get_eigenvalue( int nev ) const;

    class Iterator : public std::iterator<std::random_access_iterator_tag,
                                          EigenvalueSolver::result>
    {
      public:
        int nev;
        const EigenvalueSolver& e;

        explicit Iterator( const int i, const EigenvalueSolver& es )
            : nev( i ), e( es )
        {
        }

        Iterator( const Iterator& i ) : nev( i.nev ), e( i.e ){};

        Iterator( Iterator&& i ) : nev( i.nev ), e( i.e ) {}

        Iterator& operator=( Iterator rhs );

        Iterator& operator++();
        Iterator operator++( int );
        Iterator& operator+=( Iterator::difference_type diff );
        Iterator& operator--();
        Iterator operator--( int );
        Iterator& operator-=( Iterator::difference_type diff );

        Iterator::difference_type operator-( const Iterator& other );
        Iterator operator-( int n );
        Iterator operator+( int n );

        Iterator::value_type operator[]( int n );

        bool operator>( const Iterator& rhs );
        bool operator<( const Iterator& rhs );
        bool operator>=( const Iterator& rhs );
        bool operator<=( const Iterator& rhs );

        bool operator!=( const Iterator& rhs );
        Iterator::value_type operator*();
        std::unique_ptr<EigenvalueSolver::Iterator::value_type>
        operator->();
    };

    EigenvalueSolver::Iterator begin() const;
    EigenvalueSolver::result front();

    EigenvalueSolver::Iterator end() const;
    EigenvalueSolver::result back();

    EPS e_;
    Matrix& op_;
    Matrix* inner_product_space;
};
}
