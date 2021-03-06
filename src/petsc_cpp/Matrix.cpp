#include <petsc_cpp/Petsc.hpp>

extern "C" {
#include <petsc_cpp/HermitianTranspose.h>
}

namespace petsc
{
// assignment operator:
Matrix& Matrix::operator=( const Matrix& other )
{
    assert( other.assembled );
    if ( other.assembled && assembled ) {
        // assumption here that they have the same nonzero patter... this
        // could definitely be wrong.
        MatCopy( other.m_, m_, SAME_NONZERO_PATTERN );
    } else if ( other.assembled ) {
        MatDestroy( &m_ );
        MatConvert( other.m_, to_MatType( mat_type ), MAT_INITIAL_MATRIX, &m_ );
    }
    return *this;
}

void swap( Matrix& first, Matrix& second ) noexcept
{
    using std::swap;

    swap( first.has_type, second.has_type );
    swap( first.assembled, second.assembled );
    swap( first.owned, second.owned );
    swap( first.mat_type, second.mat_type );
    swap( first.m_, second.m_ );
}

// modifiers:
void Matrix::set_type( const MatType t )
{
    MatSetType( m_, t );
    has_type = true;
}

void Matrix::set_option( const MatOption o, bool b )
{
    MatSetOption( m_, o, b ? PETSC_TRUE : PETSC_FALSE );
}

void Matrix::set_size( const int n_global,
                       const int m_global,
                       const int n_local,
                       const int m_local,
                       const int n_block,
                       const int m_block )
{
    MatSetSizes( m_, n_local, m_local, n_global, m_global );
    if ( block_type( mat_type ) && n_block != m_block )
        MatSetBlockSizes( m_, n_block, m_block );
    else if ( block_type( mat_type ) && n_block != 1 )
        MatSetBlockSize( m_, n_block );
}

// set value:
void Matrix::set_value( const int n, const int m, PetscScalar v )
{
    MatSetValues( m_, 1, &n, 1, &m, &v, INSERT_VALUES );
    assembled = false;
}

Matrix Matrix::get_empty_matrix() const
{
    Mat new_Mat;
    MatDuplicate( m_, MAT_SHARE_NONZERO_PATTERN, &new_Mat );
    return Matrix( new_Mat );
}

// assemble!
void Matrix::assemble()
{
    MatAssemblyBegin( m_, MAT_FINAL_ASSEMBLY );
    MatAssemblyEnd( m_, MAT_FINAL_ASSEMBLY );
    assembled = true;
}

// getters:
MPI_Comm Matrix::comm() const
{
    MPI_Comm c;
    PetscObjectGetComm( (PetscObject)m_, &c );
    return c;
}

int Matrix::rank() const
{
    int rank;
    MPI_Comm_rank( comm(), &rank );
    return rank;
}

Matrix& Matrix::transpose()
{
    static Mat old_mat = m_;
    if ( old_mat == m_ )
        MatCreateTranspose( old_mat, &m_ );
    else {
        MatDestroy( &m_ );
        m_ = old_mat;
    }
    return *this;
}

Matrix& Matrix::hermitian_transpose()
{
    static Mat old_mat = m_;
    if ( old_mat == m_ )
        MatCreateHermitianTranspose( old_mat, &m_ );
    else {
        MatDestroy( &m_ );
        m_ = old_mat;
    }
    return *this;
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

Matrix& Matrix::operator*=( const std::complex<double>& alpha )
{
    MatScale( m_, alpha );
    return *this;
}
Matrix& Matrix::operator/=( const std::complex<double>& alpha )
{
    MatScale( m_, 1. / alpha );
    return *this;
}

Matrix& Matrix::operator+=( const Vector& D )
{
    MatDiagonalSet( m_, D.v_, ADD_VALUES );
    return *this;
}

Matrix& Matrix::operator+=( const std::complex<double>& alpha )
{
    MatShift( m_, alpha );
    return *this;
}

Vector Matrix::operator*( const Vector& v ) const
{
    Vector out = this->get_left_vector();
    MatMult( this->m_, v.v_, out.v_ );
    return out;
}

void Matrix::Ax( const Vector& x, Vector& b ) const
{
    MatMult( this->m_, x.v_, b.v_ );
}

Matrix& Matrix::diagonal_scale( const Vector& l, const Vector& r )
{
    MatDiagonalScale( this->m_, l.v_, r.v_ );
    return *this;
}

Matrix& Matrix::diagonal_scale( const Vector& l )
{
    MatDiagonalScale( this->m_, l.v_, NULL );
    return *this;
}

Matrix& Matrix::shallow_copy( const Matrix& a )
{
    this->m_ = a.m_;
    owned = false;
    return *this;
}

void Matrix::print() const { MatView( m_, PETSC_VIEWER_STDOUT_WORLD ); }

void Matrix::to_file( const std::string& filename ) const
{
    PetscViewer view;
    PetscViewerBinaryOpen( this->comm(), filename.c_str(), FILE_MODE_WRITE,
                           &view );
    MatView( m_, view );
    PetscViewerDestroy( &view );
}

Matrix transpose( const Matrix& A )
{
    Mat m;
    MatCreateTranspose( A.m_, &m );
    return Matrix{m};
}

Matrix hermitian_transpose( const Matrix& A )
{
    // argh, waste cycles by doing a deep version.
    Mat m;
    MatHermitianTranspose( A.m_, MAT_INITIAL_MATRIX, &m );
    return Matrix{m};
}
}
