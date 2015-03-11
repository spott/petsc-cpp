#pragma once

#include <petsc.h>
#ifdef SLEPC
#include <slepc.h>
#endif

#include <petsc_cpp/Vector.hpp>
#include <petsc_cpp/Matrix.hpp>

namespace petsc
{

class TimeStepper
{
    /* Interested in solving problems of the type:

       F(t, u, u') = G(t, u)

       Where the jacobian is F_u + (shift) F_u'.  G_u is the RHS jacobian.

       For the most common case:

       u' = A(t) u:

       G = A(t) u
       G_u = A(t).
       F = u'
       F_u + (shift) F_u' = (shift)
    */
  public:
    using RHSJacobian =
        PetscErrorCode ( * )( TS, PetscReal, Vec, Mat, Mat, void* );
    using RHS = PetscErrorCode ( * )( TS, PetscReal, Vec, Vec, void* );

    using LHSJacobian = PetscErrorCode ( * )(
        TS, PetscReal, Vec, Vec, PetscReal, Mat, Mat, void* );
    using LHS = PetscErrorCode ( * )( TS, PetscReal, Vec, Vec, Vec, void* );

    enum class problem_type { linear = TS_LINEAR, nonlinear = TS_NONLINEAR };

    struct time {
        double t0, tf, dt;
    };

    TimeStepper( problem_type type,
                 TSType solver_type,
                 const MPI_Comm& comm = PETSC_COMM_WORLD )
    {
        TSCreate( comm, &ts );
        TSSetProblemType( ts, static_cast<TSProblemType>( type ) );
        TSSetType( ts, solver_type );
    }

    /* For simple case u' = A(t) u */
    // Jac has approximate type (Vector& u, Matrix& A, Matrix& B, TimeStepper&
    // ts, double time)
    template <typename Jac>
    TimeStepper( Jac G_u,
                 Matrix& A,
                 time t,
                 TSType solver_type,
                 const MPI_Comm& comm = PETSC_COMM_WORLD )
        : TimeStepper( problem_type::linear, solver_type, comm )
    {
        assert( A.comm() == comm );
        TSSetInitialTimeStep( ts, t.t0, t.dt );
        // TSSetSolution( ts, solution.v_ );
        TSSetDuration( ts, static_cast<int>( ( t.tf - t.t0 ) / t.dt ) + 10,
                       t.tf );
        TSSetFromOptions( ts );

        TSSetRHSFunction( ts, NULL, TSComputeRHSFunctionLinear, NULL );
        TSSetRHSJacobian( ts, A.m_, A.m_,
                          TimeStepper::RHSJacobian_function<Jac>, &G_u );
    }

    template <typename Jac>
    static PetscErrorCode
    RHSJacobian_function( TS ts, double t_, Vec u, Mat A, Mat B, void* G_u )
    {
        ( *(Jac*)G_u )( Vector( u, false ), Matrix( A, false ),
                        Matrix( B, false ), TimeStepper( ts, false ), t_ );
        return 0;
    }


    explicit TimeStepper( TS& ts, bool owned = false )
        : ts( ts ), owned( owned )
    {
    }

    /* rule of 4.5 */
    TimeStepper( const TimeStepper& other ) = delete;

    TimeStepper( TimeStepper&& other ) : ts( other.ts )
    {
        other.ts = PETSC_NULL;
    }

    ~TimeStepper()
    {
        if ( !owned ) TSDestroy( &ts );
    }

    void solve( Vector& u_0 ) { TSSolve( ts, u_0.v_ ); }

    void print() const { TSView( ts, PETSC_VIEWER_STDOUT_WORLD ); }

    template <typename Mon>
    void set_monitor( Mon m )
    {
        TSMonitorSet( ts,
                      []( TS ts, int step, double t, Vec u, void* monitor ) {
                          ( *(Mon*)monitor )( TimeStepper( ts, false ), step, t,
                                              Vector( u, false ) );
                      },
                      &m, PETSC_NULL );
    }

    TS ts;
    bool owned{true};
};
}
