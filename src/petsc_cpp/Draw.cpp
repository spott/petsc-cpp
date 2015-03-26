#include <petsc_cpp/Petsc.hpp>
#include <petsc_cpp/Draw.hpp>

namespace petsc
{
unsigned _draw_window_num_ = 0;


void Draw::reset()
{
    PetscDrawLGReset( lg );
    current_point = 0;
}

void Draw::set_title( std::string title )
{
    PetscDrawSetTitle( draw, title.c_str() );
}

void Draw::draw_point( double* xvalues, double* yvalues )
{
    if ( f )
        for ( unsigned i = 0; i < dim; ++i )
            yvalues[i] = ( *f )( yvalues[i], current_point );
    PetscDrawLGAddPoint( lg, xvalues, yvalues );
    current_point++;
    PetscDrawLGDraw( lg );
}
void Draw::draw_point( double* yvalues )
{
    if ( f )
        for ( unsigned i = 0; i < dim; ++i )
            yvalues[i] = ( *f )( i, current_point );
    double* xvalues = new double[dim];
    if ( current_point < grid.size() )
        for ( unsigned i = 0u; i < dim; ++i ) xvalues[i] = grid[current_point];
    else if ( dx >= 0 )
        for ( unsigned i = 0u; i < dim; ++i ) xvalues[i] = current_point * dx;
    else
        for ( unsigned i = 0u; i < dim; ++i )
            xvalues[i] = double( current_point );
    PetscDrawLGAddPoint( lg, xvalues, yvalues );
    current_point++;
    PetscDrawLGDraw( lg );
    delete[] xvalues;
}

void Draw::set_limits( double xmin, double xmax, double ymin, double ymax )
{
    PetscDrawLGSetLimits( lg, xmin, xmax, ymin, ymax );
    PetscDrawAxisSetLimits( axis, xmin, xmax, ymin, ymax );
    PetscDrawAxisSetHoldLimits( axis, PETSC_TRUE );
}

void Draw::draw_vector( const Vector& v )
{
    assert( dim <= 2 );
    this->reset();
    VecScatter scatter;
    Vec local;
    VecScatterCreateToZero( v.v_, &scatter, &local );

    VecScatterBegin( scatter, v.v_, local, INSERT_VALUES, SCATTER_FORWARD );
    VecScatterEnd( scatter, v.v_, local, INSERT_VALUES, SCATTER_FORWARD );
    if ( !rank() ) {
        const PetscScalar* array;
        PetscInt start_, end_;
        VecGetArrayRead( local, &array );
        VecGetOwnershipRange( local, &start_, &end_ );
        size_t start = static_cast<size_t>( start_ );
        size_t end = static_cast<size_t>( end_ );
        for ( auto i = 0u; i < end - start; ++i ) {
            PetscReal x[2] = {double( i + start ), double( i + start )};
            if ( grid.size() > i ) {
                x[0] = grid[i];
                x[1] = grid[i];
            }
            PetscReal y[2] = {array[i].real(), array[i].imag()};
            if ( f ) {
                y[0] = ( *f )( array[i].real(), i );
                y[1] = ( *f )( array[i].imag(), i );
            }
            PetscDrawLGAddPoint( lg, x, y );
        }
        VecRestoreArrayRead( local, &array );
        VecDestroy( &local );
    }
    PetscDrawLGDraw( lg );
}
void Draw::draw_vector( const std::vector<Vector>& v )
{
    assert( dim % 2 == 0 );
    this->reset();
    VecScatter scatter;
    Vec local = new Vec[dim / 2];
    VecScatterCreateToZero( v[0].v_, &scatter, &local[0] );

    for ( unsigned i = 0u; i < dim / 2; ++i ) {
        VecScatterBegin( scatter, v[i].v_, local[i], INSERT_VALUES,
                         SCATTER_FORWARD );
        VecScatterEnd( scatter, v[i].v_, local[i], INSERT_VALUES,
                       SCATTER_FORWARD );
    }
    if ( !rank() ) {
        const PetscScalar** array = new PetscScalar* [dim / 2];
        PetscInt start_, end_;
        for ( unsigned i = 0u; i < dim / 2; ++i ) {
            int s, e;
            VecGetArrayRead( local, &array[i] );
            VecGetOwnershipRange( local, &s, &e );
            if ( i == 0 ) {
                start_ = s;
                end_ = e;
            } else {
                assert( s == start_ && e == end_ );
            }
        }
        size_t start = static_cast<size_t>( start_ );
        size_t end = static_cast<size_t>( end_ );
        // this is a local vector, and thus should span the vector:
        PetscReal* xarray = new double[dim];
        PetscReal* yarray = new double[dim];
        for ( auto i = 0u; i < end - start; ++i ) {
            PetscReal x = double( i );
            if ( grid.size() > i ) {
                x = grid[i];
            }
            for ( auto j = 0u; j < dim / 2; ++j ) {
                xarray[j * 2] = x;
                xarray[j * 2 + 1] = x;
                if ( f ) {
                    yarray[j * 2] = ( *f )( array[j][i].real(), i );
                    yarray[j * 2 + 1] = ( *f )( array[j][i].imag(), i );
                } else {
                    yarray[j * 2] = array[j][i].real();
                    yarray[j * 2 + 1] = array[j][i].imag();
                }
            }
            PetscDrawLGAddPoint( lg, xarray, yarray );
        }
        for ( unsigned i = 0u; i < dim / 2; ++i ) {
            VecRestoreArrayRead( local[i], &array[i] );
            VecDestroy( &local[i] );
        }
        delete[] xarray;
        delete[] yarray;
        delete[] array;
    }
    PetscDrawLGDraw( lg );
    delete[] local;
}
void Draw::set_grid( std::vector<double> v ) { grid = v; }

void Draw::set_function( std::function<PetscReal(PetscScalar, unsigned)> func )
{
    f = func;
}
int Draw::rank()
{
    int r;
    MPI_Comm_rank( comm(), &r );
    return r;
}
MPI_Comm Draw::comm()
{
    MPI_Comm c;
    PetscObjectGetComm( (PetscObject)draw, &c );
    return c;
}
}
