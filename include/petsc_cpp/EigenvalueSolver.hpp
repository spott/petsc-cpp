#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <cassert>


namespace petsc
{

class EigenvalueSolver
{
  public:
    enum class Type {
        hermitian = EPS_HEP,
        nonhermitian = EPS_NHEP,
        gen_hermitian = EPS_GHEP,
        gen_nonhermitian = EPS_GNHEP,
        pos_gen_nonhermitian = EPS_PGNHEP,
        indef_gen_nonhermitian = EPS_GHIEP
    };
    enum class Which {
        largest_mag = EPS_LARGEST_MAGNITUDE,
        smallest_mag = EPS_SMALLEST_MAGNITUDE,
        largest_real = EPS_LARGEST_REAL,
        smallest_real = EPS_SMALLEST_REAL,
        largest_imaginary = EPS_LARGEST_IMAGINARY,
        smallest_imaginary = EPS_SMALLEST_IMAGINARY,
        target_mag = EPS_TARGET_MAGNITUDE,
        target_real = EPS_TARGET_REAL,
        target_imag = EPS_TARGET_IMAGINARY,
        all = EPS_ALL,
        user = EPS_WHICH_USER
    };

    EigenvalueSolver( Matrix& A,
                      Matrix& space,
                      int dim,
                      Which which = Which::smallest_real,
                      Type type = Type::hermitian )
        : op_( A ), inner_product_space_mat( &space ),
          inner_product_space_diag( nullptr ), problem_type( type ),
          which( which )
    {
        assert( type == Type::hermitian || type == Type::nonhermitian );
        assert( space.comm() == A.comm() );
        EPSCreate( A.comm(), &e_ );
        EPSSetOperators( e_, A.m_, PETSC_NULL );
        EPSSetProblemType( e_, static_cast<EPSProblemType>( type ) );
        EPSSetDimensions( e_, dim, PETSC_DECIDE, PETSC_DECIDE );
        EPSSetWhichEigenpairs( e_, static_cast<EPSWhich>( which ) );
        EPSSetUp( e_ );
    }
    EigenvalueSolver( Matrix& A,
                      int dim,
                      Which which = Which::smallest_real,
                      Type type = Type::hermitian )
        : op_( A ), inner_product_space_mat( nullptr ),
          inner_product_space_diag( nullptr ), problem_type( type ),
          which( which )
    {
        assert( type == Type::hermitian || type == Type::nonhermitian );
        EPSCreate( A.comm(), &e_ );
        EPSSetOperators( e_, A.m_, PETSC_NULL );
        EPSSetProblemType( e_, static_cast<EPSProblemType>( type ) );
        EPSSetDimensions( e_, dim, PETSC_DECIDE, PETSC_DECIDE );
        EPSSetWhichEigenpairs( e_, static_cast<EPSWhich>( which ) );
        EPSSetUp( e_ );
    }

    EigenvalueSolver( Matrix& A )
        : EigenvalueSolver( A, 1, Which::smallest_real, Type::hermitian )
    {
    }

    // rule of 4.5...
    EigenvalueSolver( const EigenvalueSolver& other ) =
        delete; // we don't need no stinkin copy constructor...

    EigenvalueSolver( EigenvalueSolver&& other )
        : e_( other.e_ ), op_( other.op_ ), problem_type( other.problem_type ),
          which( other.which )
    {
        inner_product_space_mat = std::move( other.inner_product_space_mat );
        inner_product_space_diag = std::move( other.inner_product_space_diag );
        other.e_ = PETSC_NULL;
        other.inner_product_space_mat = nullptr;
        other.inner_product_space_diag = nullptr;
        // other.op_ = nullptr;
    }

    friend void swap( EigenvalueSolver& first,
                      EigenvalueSolver& second ); // nothrow

    ~EigenvalueSolver() { EPSDestroy( &e_ ); }

    // Getters:
    MPI_Comm comm() const;
    int rank() const;

    Matrix& op() const;
    Matrix& op( Matrix& op );

    void inner_product_space( Matrix&& B );
    void inner_product_space( Vector&& B );
    void solve();
    void set_initial_vector( Vector init );
    void balance( EPSBalance bal, int iter, double cuttoff = PETSC_DECIDE );
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

    template <typename Scalar>
    void save_basis( const std::string& filename,
                     std::function<void(Vector&)> modification =
                         functional::to_void<Vector> ) const;

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
        std::unique_ptr<EigenvalueSolver::Iterator::value_type> operator->();
    };

    EigenvalueSolver::Iterator begin() const;
    EigenvalueSolver::result front();

    EigenvalueSolver::Iterator end() const;
    EigenvalueSolver::result back();

    EPS e_;
    Matrix& op_;
    std::unique_ptr<Matrix> inner_product_space_mat;
    std::unique_ptr<Vector> inner_product_space_diag;
    Type problem_type;
    Which which;
};


template <typename Scalar>
void
EigenvalueSolver::save_basis( const std::string& filename,
                              std::function<void(Vector&)> modification ) const
{
    if ( num_converged() < 1 ) return;

    VecScatter scatter;
    Vec local;
    const PetscScalar* array;
    PetscInt start, end;

    Vector a = op().get_right_vector();
    std::vector<Scalar> temp;

    VecScatterCreateToZero( a.v_, &scatter, &local );

    for ( auto a : *this ) {
        modification( a.evector );
        VecScatterBegin( scatter, a.evector.v_, local, INSERT_VALUES,
                         SCATTER_FORWARD );
        VecScatterEnd( scatter, a.evector.v_, local, INSERT_VALUES,
                       SCATTER_FORWARD );

        VecGetOwnershipRange( local, &start, &end );
        VecGetArrayRead( local, &array );

        for ( int i = start; i < end; ++i ) {
            temp.push_back( functional::from_complex<Scalar>( array[i] ) );
        }

        VecRestoreArrayRead( local, &array );
        util::export_vector_binary( filename, temp, true );
        temp.clear();
    }
}
}
