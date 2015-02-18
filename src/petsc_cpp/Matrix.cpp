#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/Matrix.hpp>

namespace petsc
{
// assignment operator:
Matrix& Matrix::operator=( Matrix other )
{
    swap( *this, other );
    return *this;
}

// swap!
void swap( Matrix& first, Matrix& second ) // nothrow
{
    using std::swap;

    swap( first.has_type, second.has_type );
    swap( first.assembled, second.assembled );
    swap( first.m_, second.m_ );
}

// modifiers:
void Matrix::set_type( const MatType t )
{
    l.lock();
    MatSetType( m_, t );
    has_type = true;
    l.unlock();
}

void Matrix::set_option( const MatOption o, bool b )
{
    l.lock();
    MatSetOption( m_, o, b ? PETSC_TRUE : PETSC_FALSE );
    l.unlock();
}

void Matrix::set_size( const int n_global,
                       const int m_global,
                       const int n_local,
                       const int m_local )
{
    l.lock();
    MatSetSizes( m_, n_local, m_local, n_global, m_global );
    l.unlock();
}

// set value:
void Matrix::set_value( const int n, const int m, PetscScalar v )
{
    l.lock();
    MatSetValues( m_, 1, &n, 1, &m, &v, INSERT_VALUES );
    assembled = false;
    l.unlock();
}

// assemble!
void Matrix::assemble()
{
    l.lock();
    MatAssemblyBegin( m_, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( m_, MAT_FINAL_ASSEMBLY );
    assembled = true;
    l.unlock();
}

// getters:
MPI_Comm Matrix::comm() const
{
    static MPI_Comm comm = [=]() {
        MPI_Comm c;
        PetscObjectGetComm( (PetscObject)m_, &c );
        return c;
    }();
    return comm;
}

int Matrix::rank() const
{
    static int rank = [=]() {
        MPI_Comm c;
        int rank;
        PetscObjectGetComm( (PetscObject)m_, &c );
        MPI_Comm_rank( c, &rank );
        return rank;
    }();
    return rank;
}

std::array<int, 2> Matrix::n() const
{
    static std::array<int, 2> n = [=]() {
        std::array<int, 2> m;
        MatGetSize( m_, &m[0], &m[1] );
        return m;
    }();
    return n;
}

std::array<int, 2> Matrix::get_ownership_rows() const
{
    static std::array<int, 2> n = [=]() {
        std::array<int, 2> m;
        MatGetOwnershipRange( m_, &m[0], &m[1] );
        return m;
    }();
    return n;
}

std::array<Vector, 2> Matrix::get_vectors() const
{
    Vec a, b;
    MatGetVecs( m_, &a, &b );
    return std::array<Vector, 2>{{Vector{a}, Vector{b}}};
}

Vector Matrix::get_right_vector() const
{
    Vec a;
    MatGetVecs( m_, &a, PETSC_NULL );
    return Vector{a};
}

Vector Matrix::get_left_vector() const
{
    Vec a;
    MatGetVecs( m_, PETSC_NULL, &a );
    return Vector{a};
}

double Matrix::norm( NormType t )
{
    double norm;
    MatNorm( m_, t, &norm );
    return norm;
}

Vector Matrix::operator*( const Vector& v ) const
{
    Vector out = this->get_left_vector();
    MatMult( this->m_, v.v_, out.v_ );
    return out;
}

// print!
void Matrix::print() const { MatView( m_, PETSC_VIEWER_STDOUT_WORLD ); }

void Matrix::to_file( const std::string& filename ) const
{
    PetscViewer view;
    PetscViewerBinaryOpen( this->comm(), filename.c_str(), FILE_MODE_WRITE,
                           &view );
    MatView( m_, view );
    PetscViewerDestroy( &view );
}
}
