#define SLEPC
#include <stdexcept>
#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/EigenvalueSolver.hpp>
#include <slepc-private/epsimpl.h>

namespace petsc
{

void swap( EigenvalueSolver& first, EigenvalueSolver& second ) // nothrow
{
    using std::swap;

    swap( first.e_, second.e_ );
    swap( first.problem_type, second.problem_type );
    swap( first.which, second.which );
    swap( first.inner_product_space_mat, second.inner_product_space_mat );
    swap( first.inner_product_space_diag, second.inner_product_space_diag );
}


MPI_Comm EigenvalueSolver::comm() const
{
    MPI_Comm c;
    PetscObjectGetComm( (PetscObject)e_, &c );
    return c;
}

int EigenvalueSolver::rank() const
{
    MPI_Comm c;
    int rank;
    PetscObjectGetComm( (PetscObject)e_, &c );
    MPI_Comm_rank( c, &rank );
    return rank;
}


Matrix EigenvalueSolver::op() const
{
    Mat A;
    EPSGetOperators( e_, &A, NULL );
    return Matrix( A, false );
}
const Matrix& EigenvalueSolver::op( const Matrix& A )
{
    EPSSetOperators( e_, A.m_, NULL );
    // EPSSetProblemType( e_, static_cast<EPSProblemType>( problem_type ) );
    // EPSSetWhichEigenpairs( e_, static_cast<EPSWhich>( which ) );
    // EPSSetUp(e_);
    return A;
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

void EigenvalueSolver::set_initial_vectors( std::vector<Vector> init )
{
    Vec* vs = new Vec[init.size()];
    for ( auto i = 0u; i < init.size(); ++i ) vs[i] = init[i].v_;
    EPSSetInitialSpace( e_, static_cast<int>( init.size() ), vs );
    delete[] vs;
}

void EigenvalueSolver::set_deflation_space( std::vector<Vector> init )
{
    Vec* vs = new Vec[init.size()];
    for ( auto i = 0u; i < init.size(); ++i ) vs[i] = init[i].v_;
    EPSSetDeflationSpace( e_, static_cast<int>( init.size() ), vs );
    delete[] vs;
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
    if ( nev == -1 ) nev = static_cast<int>( val[0] );

    EPSSetDimensions( e_, nev, ncv, mpd );
}

unsigned int EigenvalueSolver::iteration_number() const
{
    int its;
    EPSGetIterationNumber( e_, &its );
    return static_cast<unsigned>( its );
}

std::array<unsigned, 3> EigenvalueSolver::dimensions() const
{
    std::array<int, 3> signed_out;
    EPSGetDimensions( e_, &( signed_out[0] ), &( signed_out[1] ),
                      &( signed_out[2] ) );
    std::array<unsigned, 3> out{{static_cast<unsigned>( signed_out[0] ),
                                 static_cast<unsigned>( signed_out[1] ),
                                 static_cast<unsigned>( signed_out[2] )}};
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

unsigned EigenvalueSolver::num_converged() const
{
    int nconv;
    EPSGetConverged( e_, &nconv );
    return static_cast<unsigned>( nconv );
}

void EigenvalueSolver::print() const
{
    EPSView( e_, PETSC_VIEWER_STDOUT_WORLD );
}


EigenvalueSolver::result EigenvalueSolver::get_eigenpair( unsigned nev ) const
{
    assert( nev < num_converged() );
    result r{nev, 0.0, op().get_right_vector()};
    int nev_signed = static_cast<int>( nev );
    EPSGetEigenpair( e_, nev_signed, &( r.evalue ), PETSC_NULL, r.evector.v_,
                     PETSC_NULL );
    // r.evector.normalize_sign();
    // if ( inner_product_space_mat != nullptr )
    //     r.evector /= std::sqrt(
    //         inner_product( r.evector, *inner_product_space_mat, r.evector )
    //         );
    // else if ( inner_product_space_diag != nullptr )
    //     r.evector /= std::sqrt(
    //         inner_product( r.evector, *inner_product_space_diag, r.evector )
    //         );
    // else
    //     std::cerr << "didn't renormalize eigenpair " << r.nev << std::endl;
    return r;
}

PetscScalar EigenvalueSolver::get_eigenvalue( unsigned nev ) const
{
    assert( nev < num_converged() );
    PetscScalar ev;
    EPSGetEigenvalue( e_, static_cast<int>( nev ), &ev, PETSC_NULL );
    return ev;
}

Vector EigenvalueSolver::get_eigenvector( unsigned nev )
{
    assert( nev < num_converged() );


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
    EPSCheckSolved( e_, 1 );
#pragma clang diagnostic pop

    if ( e_->state == EPS_STATE_SOLVED && e_->ops->computevectors )
        ( *e_->ops->computevectors )( e_ );
    e_->state = EPS_STATE_EIGENVECTORS;

    PetscInt k;
    Vec v;

    if ( !e_->perm )
        k = static_cast<int>( nev );
    else
        k = e_->perm[nev];
    BVGetColumn( ( e_->V ), k, &v );
    return Vector(
        v, [ bv = e_->V, k ]( Vec vv ) { BVRestoreColumn( bv, k, &vv ); } );
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::operator++()
{
    // assert( nev < e.num_converged()-1 );
    nev++;
    return *this;
}

EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator++( int )
{
    auto ret = *this;
    nev++;
    return ret;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator+=( EigenvalueSolver::Iterator::difference_type diff )
{
    assert( diff > 0 || ( static_cast<difference_type>( nev ) >= -diff ) );
    nev += diff;
    return *this;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::operator--()
{
    assert( nev > 0 );
    nev--;
    return *this;
}
EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator--( int )
{
    assert( nev > 0 );
    auto ret = *this;
    nev--;
    return ret;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator-=( EigenvalueSolver::Iterator::difference_type diff )
{
    assert( nev - diff < e.num_converged() && nev >= diff );
    nev -= diff;
    return *this;
}

EigenvalueSolver::Iterator::difference_type EigenvalueSolver::Iterator::
operator-( const EigenvalueSolver::Iterator& other )
{
    assert( nev >= other.nev );
    return nev - other.nev;
}

EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator-( int n )
{
    assert( ( n > 0 && static_cast<int>( nev ) >= n ) || n < 0 );
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
operator[]( unsigned n )
{
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
    if ( nev >= e.num_converged() )
        throw std::out_of_range( "EigenvalueSolver::Iterator: attempting "
                                 "to dereference an iterator that is out "
                                 "of "
                                 "range" );
    return e.get_eigenpair( nev );
}
std::unique_ptr<EigenvalueSolver::Iterator::value_type>
    EigenvalueSolver::Iterator::operator->()
{
    if ( nev >= e.num_converged() )
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
    assert( num_converged() >= 1 );
    return get_eigenpair( num_converged() - 1 );
}
}
