#pragma once

#include <vector>
#include <experimental/optional>

namespace petsc
{

extern unsigned _draw_window_num_;

struct Draw {
    Draw( unsigned dim_, const MPI_Comm comm ) : dim( dim_ )
    {
        // static unsigned w{0};
        window = _draw_window_num_;
        _draw_window_num_ += 1;
        viewer = PETSC_VIEWER_DRAW_( comm );
        PetscViewerDrawGetDraw( viewer, static_cast<int>( window ), &draw );
        PetscViewerDrawGetDrawLG( viewer, static_cast<int>( window ), &lg );
        PetscDrawSetDoubleBuffer( draw );
        PetscDrawLGSetDimension( lg, dim );
        PetscDrawLGGetAxis( lg, &axis );
    }

    Draw( double xmin,
          double xmax,
          double ymin,
          double ymax,
          unsigned dim = 2,
          const MPI_Comm comm = PETSC_COMM_WORLD )
        : Draw( dim, comm )
    {
        set_limits( xmin, xmax, ymin, ymax );
    }

    void reset();

    void set_title( std::string title );

    void draw_point( double* xvalues, double* yvalues );

    void draw_point( double* yvalues );

    void set_limits( double xmin, double xmax, double ymin, double ymax );

    void draw_vector( const Vector& v );

    void draw_vector( const std::vector<Vector>& v );

    void set_grid( std::vector<double> v );

    void set_function( std::function<PetscReal(PetscScalar, unsigned)> func );

    int rank();
    MPI_Comm comm();

  private:
    unsigned dim;
    unsigned window;
    PetscViewer viewer;
    PetscDraw draw;
    PetscDrawLG lg;
    PetscDrawAxis axis;
    unsigned current_point{0};
    std::vector<double> grid;
    double dx{0};
    std::experimental::optional<std::function<PetscReal(PetscScalar, unsigned)>>
        f;
};
}
