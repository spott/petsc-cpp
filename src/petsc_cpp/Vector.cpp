#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/Vector.hpp>

namespace petsc
{
void swap( Vector& first, Vector& second ) // nothrow
{
    using std::swap;

    swap( first.has_type, second.has_type );
    swap( first.v_, second.v_ );
}

/*************
//modifiers:
*************/

// set value:
void Vector::set_value( const int n, PetscScalar v )
{
    l.lock();
    VecSetValue( v_, n, v, INSERT_VALUES );
    assembled = false;
    l.unlock();
}

// assemble:
void Vector::assemble()
{
    l.lock();
    VecAssemblyBegin( v_ );
    VecAssemblyEnd( v_ );
    assembled = true;
    l.unlock();
}

/*************
// calculations:
*************/
double Vector::norm( NormType nt ) const
{
    double norm;
    VecNorm( v_, nt, &norm );
    return norm;
}

double Vector::normalize()
{
    double norm;
    VecNormalize( v_, &norm );
    return norm;
}

/*************
//getters:
*************/
MPI_Comm Vector::comm() const
{
    static MPI_Comm comm = [=]() {
        MPI_Comm c;
        PetscObjectGetComm( (PetscObject)v_, &c );
        return c;
    }();
    return comm;
}

size_t Vector::size() const
{
    static int n = [=]() {
        int i;
        VecGetSize( v_, &i );
        return i;
    }();
    return static_cast<size_t>( n );
}

std::array<int, 2> Vector::get_ownership_rows() const
{
    static std::array<int, 2> n = [=]() {
        std::array<int, 2> m;
        VecGetOwnershipRange( v_, &m[0], &m[1] );
        return m;
    }();
    return n;
}

void Vector::print() const { VecView( v_, PETSC_VIEWER_STDOUT_WORLD ); }

void Vector::to_file( const std::string& filename ) const
{
    PetscViewer view;
    PetscViewerBinaryOpen( this->comm(), filename.c_str(), FILE_MODE_WRITE,
                           &view );
    VecView( v_, view );
    PetscViewerDestroy( &view );
}
}
