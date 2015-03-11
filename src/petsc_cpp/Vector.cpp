#include <petsc_cpp/Petsc.hpp>

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
Vector Vector::operator/( const PetscScalar& other ) const
{
    if ( this->rank() == 0 )
        std::cerr << "using operator-: this is inefficient!!!" << std::endl;
    Vector v{*this};
    VecScale( v.v_, 1. / other );
    return v;
}
/*************
//getters:
*************/
int Vector::rank() const
{
    static int rank = [=]() {
        MPI_Comm c;
        PetscObjectGetComm( (PetscObject)v_, &c );
        int r;
        MPI_Comm_rank( c, &r );
        return r;
    }();
    return rank;
}
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


        PetscViewerDrawGetDraw( viewer, i, &draw );
        PetscViewerDrawGetDrawLG( viewer, i, &lg );
        PetscDrawSetTitle( draw, ( "Vector " + std::to_string( i ) ).c_str() );
        PetscDrawSetDoubleBuffer( draw );
        PetscDrawLGSetDimension( lg, 2 );
        PetscDrawLGReset( lg );
        if ( v != nullptr )
            PetscDrawLGSetLimits( lg, v->front(), v->back(), -1, 1 );
        else
            PetscDrawLGSetLimits( lg, 0, vectors[i].size(), -1, 1 );
        const PetscScalar* array;
        PetscInt start, end;
        // TODO: Need to scatter array first
        VecScatterBegin( scatter, vectors[i].v_, local, INSERT_VALUES,
                         SCATTER_FORWARD );
        VecScatterEnd( scatter, vectors[i].v_, local, INSERT_VALUES,
                       SCATTER_FORWARD );
        VecGetArrayRead( local, &array );
        if ( !vectors[0].rank() ) {
            VecGetOwnershipRange( local, &start, &end );

            for ( int i = 0; i < end - start; ++i ) {
                PetscReal x[2] = {double( i + start ), double( i + start )};
                if ( v != nullptr ) {
                    x[0] = ( *v )[i];
                    x[1] = ( *v )[i];
                }
                PetscReal y[2] = {array[i].real(), 10. * array[i].imag()};
                PetscDrawLGAddPoint( lg, x, y );
            }
            VecRestoreArrayRead( vectors[i].v_, &array );
        }

        double p;
        PetscDrawGetPause( draw, &p );
        if ( i < vectors.size() - 1 ) PetscDrawSetPause( draw, 0 );
        PetscDrawLGDraw( lg );
        PetscDrawSetPause( draw, p );
    }
}

void Vector::draw( const std::vector<double>* v ) const
{
    static PetscViewer viewer = PETSC_VIEWER_DRAW_( this->comm() );
    PetscDraw draw;
    PetscDrawLG lg;

    PetscViewerDrawGetDraw( viewer, 0, &draw );
    PetscViewerDrawGetDrawLG( viewer, 0, &lg );
    PetscDrawSetTitle( draw, "Vector" );
    PetscDrawSetDoubleBuffer( draw );
    PetscDrawLGSetDimension( lg, 2 );
    PetscDrawLGReset( lg );
    if ( v != nullptr )
        PetscDrawLGSetLimits( lg, v->front(), v->back(), -1, 1 );
    else
        PetscDrawLGSetLimits( lg, 0, this->size(), -1, 1 );

    const PetscScalar* array;
    PetscInt start, end;
    // TODO: Need to scatter array first
    VecGetArrayRead( v_, &array );
    VecGetOwnershipRange( v_, &start, &end );

    for ( int i = 0; i < end - start; ++i ) {
        PetscReal x[2] = {double( i + start ), double( i + start )};
        if ( v != nullptr ) {
            x[0] = ( *v )[i];
            x[1] = ( *v )[i];
        }
        PetscReal y[2] = {array[i].real(), array[i].imag()};
        PetscDrawLGAddPoint( lg, x, y );
    }
    VecRestoreArrayRead( v_, &array );
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
}
