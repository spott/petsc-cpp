#define SLEPC

#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/EigenvalueSolver.hpp>


namespace petsc
{

void swap( EigenvalueSolver& first, EigenvalueSolver& second ) // nothrow
{
    using std::swap;

    swap( first.e_, second.e_ );
    swap( first.op_, second.op_ );
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


Matrix EigenvalueSolver::op() const { return op_; }

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


EigenvalueSolver::result EigenvalueSolver::get_eigenpair( int nev )
{
    auto v = op().get_right_vector();
    PetscScalar ev;
    EPSGetEigenpair( e_, nev, &ev, PETSC_NULL, v.v_, PETSC_NULL );
    return result{nev, ev, v};
}


EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::
operator=( Iterator rhs )
{
    *this = EigenvalueSolver::Iterator( rhs.nev, rhs.e );
    return *this;
}

EigenvalueSolver::Iterator& EigenvalueSolver::Iterator::operator++()
{
    nev++;
    return *this;
}

EigenvalueSolver::Iterator EigenvalueSolver::Iterator::operator++( int )
{
    auto ret = *this;
    nev++;
    return ret;
}

bool EigenvalueSolver::Iterator::operator!=( const Iterator& rhs )
{
    return ( rhs.nev != nev || ( &( rhs.e ) != &( e ) ) );
}
EigenvalueSolver::Iterator::value_type EigenvalueSolver::Iterator::
operator*()
{
    return e.get_eigenpair( nev );
}

EigenvalueSolver::Iterator EigenvalueSolver::begin()
{
    return EigenvalueSolver::Iterator( 0, *this );
}
EigenvalueSolver::Iterator EigenvalueSolver::end()
{
    return EigenvalueSolver::Iterator( num_converged() - 1, *this );
}
}
