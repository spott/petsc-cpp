#define SLEPC

#include <stdexcept>
#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/EigenvalueSolver.hpp>


namespace petsc
{

void swap( EigenvalueSolver& first, EigenvalueSolver& second ) // nothrow
{
    using std::swap;

    swap( first.e_, second.e_ );
    swap( first.op_, second.op_ );
    swap( first.problem_type, second.problem_type );
    swap( first.which, second.which );
    swap( first.inner_product_space_mat, second.inner_product_space_mat );
    swap( first.inner_product_space_diag, second.inner_product_space_diag );
}


MPI_Comm EigenvalueSolver::comm() const
{
    static MPI_Comm comm = [=]() {
        MPI_Comm c;
        PetscObjectGetComm( (PetscObject)e_, &c );
        return c;
    }();
    return comm;
}


Matrix& EigenvalueSolver::op() const { return op_; }
Matrix& EigenvalueSolver::op( Matrix& A )
{
    EPSSetOperators( e_, A.m_, NULL );
    // EPSSetProblemType( e_, static_cast<EPSProblemType>( problem_type ) );
    // EPSSetWhichEigenpairs( e_, static_cast<EPSWhich>( which ) );
    // EPSSetUp(e_);
    assert( &A == &op_ );
    return op_;
}

void EigenvalueSolver::inner_product_space( Matrix&& B )
{
    using namespace std;
    assert( inner_product_space_diag == nullptr );
    inner_product_space_mat = make_unique<Matrix>( move( B ) );
}
void EigenvalueSolver::inner_product_space( Vector&& B )
{
    using namespace std;
    assert( inner_product_space_mat == nullptr );
    inner_product_space_diag = make_unique<Vector>( move( B ) );
}
void EigenvalueSolver::solve()
{
    static bool set_from_options = false;
    if ( !set_from_options ) {
        EPSSetFromOptions( e_ );
        set_from_options = true;
    }

    EPSSolve( e_ );
}

void EigenvalueSolver::set_initial_vector( Vector init )
{
    EPSSetInitialSpace( e_, 1, &init.v_ );
}

void EigenvalueSolver::balance( EPSBalance bal, int iter, double cutoff )
{
    EPSSetBalance( e_, bal, iter, cutoff );
}

void EigenvalueSolver::shift_invert( std::complex<double> sigma )
{
    ST st;
    EPSGetST( e_, &st );
    STSetShift( st, sigma );
    STSetType( st, STSINVERT );
}

void EigenvalueSolver::dimensions( int nev, int mpd, int ncv )
{
    auto val = dimensions();
    if ( mpd == -1 ) mpd = PETSC_DECIDE;
    if ( ncv == -1 ) ncv = PETSC_DECIDE;
    if ( nev == -1 ) nev = val[0];

    EPSSetDimensions( e_, nev, ncv, mpd );
}

int EigenvalueSolver::iteration_number() const
{
    int its;
    EPSGetIterationNumber( e_, &its );
    return its;
}

std::array<int, 3> EigenvalueSolver::dimensions() const
{
    std::array<int, 3> out;
    EPSGetDimensions( e_, &( out[0] ), &( out[1] ), &( out[2] ) );
    return out;
}

std::tuple<PetscReal, PetscInt> EigenvalueSolver::tolerances() const
{
    PetscReal tol;
    PetscInt its;
    EPSGetTolerances( e_, &tol, &its );
    return std::tie( tol, its );
}

void EigenvalueSolver::tolerances( double tol, int its )
{
    EPSSetTolerances( e_, tol, its );
}

int EigenvalueSolver::num_converged() const
{
    int nconv;
    EPSGetConverged( e_, &nconv );
    return nconv;
}

void EigenvalueSolver::print() const
{
    EPSView( e_, PETSC_VIEWER_STDOUT_WORLD );
}

void EigenvalueSolver::save_basis( const std::string& filename ) const
{
    if ( num_converged() < 1 ) return;

    VecScatter scatter;
    Vec local;
    const PetscScalar* array;
    PetscInt start, end;

    Vector a = op().get_right_vector();
    std::vector<std::complex<double>> temp;

    VecScatterCreateToZero( a.v_, &scatter, &local );

    for ( auto a : *this ) {
        VecScatterBegin( scatter, a.evector.v_, local, INSERT_VALUES,
                         SCATTER_FORWARD );
        VecScatterEnd( scatter, a.evector.v_, local, INSERT_VALUES,
                       SCATTER_FORWARD );

        VecGetOwnershipRange( local, &start, &end );
        VecGetArrayRead( local, &array );

        for ( int i = start; i < end; ++i ) {
            temp.push_back( array[i] );
        }

        VecRestoreArrayRead( local, &array );
        util::export_vector_binary( filename, temp, true );
        temp.clear();
    }
}
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
    std::vector<std::complex<double>> temp;

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
            temp.push_back( array[i] );
        }

        VecRestoreArrayRead( local, &array );
        util::export_vector_binary( filename, temp, true );
        temp.clear();
    }
}


EigenvalueSolver::result EigenvalueSolver::get_eigenpair( int nev ) const
{
    assert( nev < num_converged() );
    auto v = op().get_right_vector();
    PetscScalar ev;
    EPSGetEigenpair( e_, nev, &ev, PETSC_NULL, v.v_, PETSC_NULL );
    v.normalize_sign();
    if ( inner_product_space_mat != nullptr )
        v /= std::sqrt( inner_product( v, *inner_product_space_mat, v ) );
    else if ( inner_product_space_diag != nullptr )
        v /= std::sqrt( inner_product( v, *inner_product_space_diag, v ) );
    return result{nev, ev, v};
}

PetscScalar EigenvalueSolver::get_eigenvalue( int nev ) const
{
    assert( nev < num_converged() );
    PetscScalar ev;
    EPSGetEigenvalue( e_, nev, &ev, PETSC_NULL );
    return ev;
}


EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator=( Iterator rhs )
{
    *this = EigenvalueSolver::Iterator( rhs.nev, rhs.e );
    return *this;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::operator++()
{
    // assert( nev < e.num_converged()-1 );
    nev++;
    return *this;
}

EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator++( int )
{
    // assert( nev < e.num_converged()-1 );
    auto ret = *this;
    nev++;
    return ret;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator+=( EigenvalueSolver::Iterator::difference_type diff )
{

    // assert( nev + diff < e.num_converged() && nev + diff >= 0 );
    nev += diff;
    return *this;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::operator--()
{
    // assert( nev > 0 );
    nev--;
    return *this;
}
EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator--( int )
{
    // assert( nev > 0 );
    auto ret = *this;
    nev--;
    return ret;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator-=( EigenvalueSolver::Iterator::difference_type diff )
{
    // assert( nev - diff < e.num_converged() && nev - diff >= 0 );
    nev -= diff;
    return *this;
}

EigenvalueSolver::Iterator::difference_type EigenvalueSolver::Iterator::
operator-( const EigenvalueSolver::Iterator& other )
{
    return nev - other.nev;
}

EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator-( int n )
{
    EigenvalueSolver::Iterator ret{*this};
    ret -= n;
    return ret;
}
EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator+( int n )
{
    EigenvalueSolver::Iterator ret{*this};
    ret += n;
    return ret;
}

EigenvalueSolver::Iterator::value_type EigenvalueSolver::Iterator::
operator[]( int n )
{
    // assert( n < e.num_converged() && n >= 0 );
    nev = n;
    return e.get_eigenpair( nev );
}

bool EigenvalueSolver::Iterator::
operator!=( const EigenvalueSolver::Iterator& rhs )
{
    return ( rhs.nev != nev || ( &( rhs.e ) != &( e ) ) );
}

bool EigenvalueSolver::Iterator::
operator>( const EigenvalueSolver::Iterator& rhs )
{
    return nev > rhs.nev;
}
bool EigenvalueSolver::Iterator::
operator<( const EigenvalueSolver::Iterator& rhs )
{
    return nev < rhs.nev;
}
bool EigenvalueSolver::Iterator::
operator>=( const EigenvalueSolver::Iterator& rhs )
{
    return nev >= rhs.nev;
}
bool EigenvalueSolver::Iterator::
operator<=( const EigenvalueSolver::Iterator& rhs )
{
    return nev <= rhs.nev;
}

EigenvalueSolver::Iterator::value_type EigenvalueSolver::Iterator::operator*()
{
    if ( nev < 0 || nev >= e.num_converged() )
        throw std::out_of_range( "EigenvalueSolver::Iterator: attempting "
                                 "to dereference an iterator that is out "
                                 "of "
                                 "range" );
    return e.get_eigenpair( nev );
}
std::unique_ptr<EigenvalueSolver::Iterator::value_type>
    EigenvalueSolver::Iterator::operator->()
{
    if ( nev < 0 || nev >= e.num_converged() )
        throw std::out_of_range( "EigenvalueSolver::Iterator: attempting "
                                 "to dereference an iterator "
                                 "that is out of range" );
    return std::make_unique<EigenvalueSolver::result>( e.get_eigenpair( nev ) );
}

EigenvalueSolver::Iterator EigenvalueSolver::begin() const
{
    return EigenvalueSolver::Iterator( 0, *this );
}
EigenvalueSolver::Iterator EigenvalueSolver::end() const
{
    return EigenvalueSolver::Iterator( num_converged(), *this );
}

EigenvalueSolver::result EigenvalueSolver::front()
{
    return get_eigenpair( 0 );
}
EigenvalueSolver::result EigenvalueSolver::back()
{
    return get_eigenpair( num_converged() - 1 );
}
}
