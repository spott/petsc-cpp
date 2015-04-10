#include <petsc_cpp/Petsc.hpp>

namespace petsc
{
void swap( Vector& first, Vector& second ) // nothrow
{
    using std::swap;

    swap( first.v_, second.v_ );
}


/*************
//modifiers:
*************/

// set value:
void Vector::set_value( const int n, PetscScalar v )
{
    VecSetValue( v_, n, v, INSERT_VALUES );
}
Vector& Vector::set_all( const PetscScalar v )
{
    VecSet( v_, v );
    return *this;
}

// assemble:
void Vector::assemble()
{
    VecAssemblyBegin( v_ );
    VecAssemblyEnd( v_ );
}

Vector& Vector::conjugate()
{
    VecConjugate( v_ );
    return *this;
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

std::complex<double> Vector::normalize_sign()
{
    std::complex<double> sign, *xx;
    if ( !this->rank() ) {
        auto r = this->get_ownership_rows();
        VecGetArray( v_, &xx );
        auto x = *xx;
        for ( int i = 1; i < r[1] - r[0]; ++i )
            x = std::abs( x ) > std::abs( xx[i] ) ? x : xx[i];
        sign = x / std::abs( x );
        VecRestoreArray( v_, &xx );
    }
    MPI_Bcast( &sign, 1, MPIU_SCALAR, 0, this->comm() );
    VecScale( v_, 1.0 / sign );
    return sign;
}

Vector Vector::operator-( Vector other ) const
{
    if ( this->rank() == 0 )
        std::cerr << "using operator-: this is inefficient!!!" << std::endl;
    VecAYPX( other.v_, -1.0, this->v_ );
    return other;
}
Vector Vector::operator+( Vector other ) const
{
    if ( this->rank() == 0 )
        std::cerr << "using operator+: this is inefficient!!!" << std::endl;
    VecAYPX( other.v_, 1.0, this->v_ );
    return other;
}
Vector Vector::operator*( const PetscScalar& other ) const
{
    if ( this->rank() == 0 )
        std::cerr << "using operator*: this is inefficient!!!" << std::endl;
    Vector v{*this};
    VecScale( v.v_, other );
    return v;
}

Vector& Vector::operator/=( const PetscScalar& other )
{
    VecScale( v_, 1. / other );
    return *this;
}
Vector& Vector::operator*=( const PetscScalar& other )
{
    VecScale( v_, other );
    return *this;
}
Vector& Vector::operator*=( const Vector& other )
{
    VecPointwiseMult( v_, v_, other.v_ );
    return *this;
}
Vector Vector::operator/( const PetscScalar& other ) const
{
    if ( this->rank() == 0 )
        std::cerr << "using operator-: this is inefficient!!!" << std::endl;
    Vector v{*this};
    VecScale( v.v_, 1. / other );
    return v;
}
Vector& Vector::operator+=( const Vector& other )
{
    VecAXPY( v_, 1, other.v_ );
    return *this;
}
Vector& Vector::operator-=( const Vector& other )
{
    VecAXPY( v_, -1, other.v_ );
    return *this;
}
Vector& Vector::axpy( const PetscScalar& alpha, const Vector& x )
{
    VecAXPY( v_, alpha, x.v_ );
    return *this;
}
/*************
//getters:
*************/
int Vector::rank() const
{
    int r;
    MPI_Comm_rank( comm(), &r );
    return r;
}
MPI_Comm Vector::comm() const
{
    MPI_Comm c;
    PetscObjectGetComm( (PetscObject)v_, &c );
    return c;
}

size_t Vector::size() const
{
    int i;
    VecGetSize( v_, &i );
    return static_cast<size_t>( i );
}

std::array<int, 2> Vector::get_ownership_rows() const
{
    std::array<int, 2> m;
    VecGetOwnershipRange( v_, &m[0], &m[1] );
    return m;
}

Vector Vector::duplicate() const
{
    Vec n;
    VecDuplicate( this->v_, &n );
    return Vector{n};
}

void Vector::print() const { VecView( v_, PETSC_VIEWER_STDOUT_WORLD ); }

void Vector::draw( const std::vector<Vector>& vectors,
                   const std::vector<double>* v )
{
    static PetscViewer viewer = PETSC_VIEWER_DRAW_( vectors[0].comm() );
    VecScatter scatter;
    Vec local;
    VecScatterCreateToZero( vectors[0].v_, &scatter, &local );
    for ( auto i = 0u; i < vectors.size(); ++i ) {
        PetscDraw draw;
        PetscDrawLG lg;


        PetscViewerDrawGetDraw( viewer, static_cast<int>( i ), &draw );
        PetscViewerDrawGetDrawLG( viewer, static_cast<int>( i ), &lg );
        PetscDrawSetTitle( draw, ( "Vector " + std::to_string( i ) ).c_str() );
        PetscDrawSetDoubleBuffer( draw );
        PetscDrawLGSetDimension( lg, 2 );
        PetscDrawLGReset( lg );

        // max & min
        PetscReal min, max;
        VecMax( vectors[i].v_, PETSC_NULL, &max );
        VecMin( vectors[i].v_, PETSC_NULL, &min );
        if ( v != nullptr )
            PetscDrawLGSetLimits( lg, v->front(), v->back(), -min, max );
        else
            PetscDrawLGSetLimits( lg, 0, vectors[i].size(), -min, max );
        const PetscScalar* array;
        PetscInt start_, end_;
        VecScatterBegin( scatter, vectors[i].v_, local, INSERT_VALUES,
                         SCATTER_FORWARD );
        VecScatterEnd( scatter, vectors[i].v_, local, INSERT_VALUES,
                       SCATTER_FORWARD );
        VecGetArrayRead( local, &array );
        if ( !vectors[0].rank() ) {
            VecGetOwnershipRange( local, &start_, &end_ );
            size_t start = static_cast<size_t>( start_ );
            size_t end = static_cast<size_t>( end_ );

            for ( auto j = 0u; j < end - start; ++j ) {
                PetscReal x[2] = {double( j + start ), double( j + start )};
                if ( v != nullptr ) {
                    x[0] = ( *v )[j];
                    x[1] = ( *v )[j];
                }
                PetscReal y[2] = {array[j].real(), array[j].imag()};
                PetscDrawLGAddPoint( lg, x, y );
            }
        }
        VecRestoreArrayRead( local, &array );

        double p;
        PetscDrawGetPause( draw, &p );
        if ( i < vectors.size() - 1 ) PetscDrawSetPause( draw, 0 );
        PetscDrawLGDraw( lg );
        PetscDrawSetPause( draw, p );
    }
    VecDestroy( &local );
}

void Vector::draw( const std::vector<double>* v ) const
{
    static PetscViewer viewer = PETSC_VIEWER_DRAW_( this->comm() );
    PetscDraw draw;
    PetscDrawLG lg;
    VecScatter scatter;
    Vec local;
    VecScatterCreateToZero( v_, &scatter, &local );

    PetscViewerDrawGetDraw( viewer, 0, &draw );
    PetscViewerDrawGetDrawLG( viewer, 0, &lg );
    PetscDrawSetTitle( draw, "Vector" );
    PetscDrawSetDoubleBuffer( draw );
    PetscDrawLGSetDimension( lg, 2 );
    PetscDrawLGReset( lg );
    PetscReal min, max;
    VecMax( v_, PETSC_NULL, &max );
    VecMin( v_, PETSC_NULL, &min );
    if ( v != nullptr ) {
        PetscDrawLGSetLimits( lg, v->front(), v->back(), -min, max );
        assert( v->size() >= size() );
    } else
        PetscDrawLGSetLimits( lg, 0, this->size(), -min, max );

    const PetscScalar* array;
    PetscInt start_, end_;
    VecScatterBegin( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    VecScatterEnd( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    if ( !rank() ) {
        VecGetArrayRead( local, &array );
        VecGetOwnershipRange( local, &start_, &end_ );
        size_t start = static_cast<size_t>( start_ );
        size_t end = static_cast<size_t>( end_ );
        // this is a local vector, and thus should span the vector:
        assert( start == 0 );
        assert( end == size() );
        if ( v != nullptr ) end = v->size();

        for ( auto i = 0u; i < end - start; ++i ) {
            PetscReal x[2] = {double( i + start ), double( i + start )};
            if ( v != nullptr ) {
                x[0] = ( *v )[i];
                x[1] = ( *v )[i];
            }
            PetscReal y[2] = {array[i].real(), array[i].imag()};
            PetscDrawLGAddPoint( lg, x, y );
        }
        VecRestoreArrayRead( local, &array );
        VecDestroy( &local );
    }
    PetscDrawLGDraw( lg );
}

void Vector::draw( std::function<double(double)> f,
                   const std::vector<double>* v ) const
{
    static PetscViewer viewer = PETSC_VIEWER_DRAW_( this->comm() );
    PetscDraw draw;
    PetscDrawLG lg;
    VecScatter scatter;
    Vec local;
    VecScatterCreateToZero( v_, &scatter, &local );

    PetscViewerDrawGetDraw( viewer, 0, &draw );
    PetscViewerDrawGetDrawLG( viewer, 0, &lg );
    PetscDrawSetTitle( draw, "Vector" );
    PetscDrawSetDoubleBuffer( draw );
    PetscDrawLGSetDimension( lg, 2 );
    PetscDrawLGReset( lg );
    PetscReal min, max;
    VecMax( v_, PETSC_NULL, &max );
    VecMin( v_, PETSC_NULL, &min );
    if ( v != nullptr ) {
        PetscDrawLGSetLimits( lg, v->front(), v->back(), f( min ), f( max ) );
        assert( v->size() >= size() );
    } else
        PetscDrawLGSetLimits( lg, 0, this->size(), f( min ), f( max ) );

    const PetscScalar* array;
    PetscInt start_, end_;
    // TODO: Need to scatter array first
    VecScatterBegin( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    VecScatterEnd( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    if ( !rank() ) {
        VecGetArrayRead( local, &array );
        VecGetOwnershipRange( local, &start_, &end_ );
        size_t start = static_cast<size_t>( start_ );
        size_t end = static_cast<size_t>( end_ );
        // this is a local vector, and thus should span the vector:
        assert( start == 0 );
        assert( end == size() );
        if ( v != nullptr ) end = v->size();

        for ( auto i = 0u; i < end - start; ++i ) {
            PetscReal x[2] = {double( i + start ), double( i + start )};
            if ( v != nullptr ) {
                x[0] = ( *v )[i];
                x[1] = ( *v )[i];
            }
            PetscReal y[2] = {f( array[i].real() ), f( array[i].imag() )};
            PetscDrawLGAddPoint( lg, x, y );
        }
        VecRestoreArrayRead( local, &array );
        VecDestroy( &local );
    }
    PetscDrawLGDraw( lg );
}


void Vector::to_file( const std::string& filename ) const
{
    PetscViewer view;
    PetscViewerBinaryOpen( this->comm(), filename.c_str(), FILE_MODE_WRITE,
                           &view );
    VecView( v_, view );
    PetscViewerDestroy( &view );
}

std::vector<std::complex<double>> Vector::to_vector() const
{
    VecScatter scatter;
    Vec local;
    const PetscScalar* array;
    PetscInt start_, end_;

    std::vector<std::complex<double>> out;
    VecScatterCreateToZero( v_, &scatter, &local );
    VecScatterBegin( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    VecScatterEnd( scatter, v_, local, INSERT_VALUES, SCATTER_FORWARD );
    VecGetArrayRead( local, &array );
    VecGetOwnershipRange( local, &start_, &end_ );

    for ( auto a = 0u; a < static_cast<unsigned>( end_ - start_ ); a++ ) {
        out.push_back( array[a] );
    }

    VecRestoreArrayRead( local, &array );
    VecDestroy( &local );

    return out;
}

Vector operator*( const PetscScalar& alpha, Vector b )
{
    if ( b.rank() == 0 )
        std::cerr << "using operator*: this is inefficient!!!" << std::endl;
    VecScale( b.v_, alpha );
    return b;
}

Vector conjugate( Vector v )
{
    VecConjugate( v.v_ );
    return v;
}

Vector abs_square( Vector v )
{
    VecAbs( v.v_ );
    VecPow( v.v_, 2 );
    return v;
}
}
