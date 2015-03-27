#pragma once

#include <petsc_cpp/Petsc.hpp>

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

    struct times {
        double ti, tf, dt;
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
    TimeStepper( Jac G_u, Matrix& A, times t, TSType solver_type)
        : TimeStepper( problem_type::linear, solver_type, A.comm() )
    {
        static Jac G_u_ = G_u;
        TSSetInitialTimeStep( ts, t.ti, t.dt );
        TSSetDuration( ts, static_cast<int>( ( t.tf - t.ti ) / t.dt ) + 10,
                       t.tf );
        TSSetRHSFunction( ts, NULL, TSComputeRHSFunctionLinear, NULL );
        TSSetRHSJacobian( ts, A.m_, A.m_,
                          TimeStepper::RHSJacobian_function<Jac>, &G_u_ );
    }

    template <typename Jac, typename RHS>
    TimeStepper( Jac G_u, RHS G, Matrix& A, times t, TSType solver_type)
        : TimeStepper( problem_type::nonlinear, solver_type, A.comm() )
    {
        static Jac G_u_ = G_u;
        static RHS G_ = G;
        TSSetInitialTimeStep( ts, t.ti, t.dt );
        TSSetDuration( ts, static_cast<int>( ( t.tf - t.ti ) / t.dt ) + 10,
                       t.tf );
        TSSetRHSFunction( ts, NULL, TimeStepper::RHS_function<RHS>, &G_ );

        TSSetRHSJacobian( ts, A.m_, A.m_,
                          TimeStepper::RHSJacobian_function<Jac>, &G_u_ );
    }


    explicit TimeStepper( TS& ts_, bool owned_ = false )
        : ts( ts_ ), owned( owned_ )
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
        if ( owned ) TSDestroy( &ts );
    }

    void solve( Vector& u_0 ) {
        TSSetFromOptions( ts );
        TSSolve( ts, u_0.v_ ); }

    void print() const { TSView( ts, PETSC_VIEWER_STDOUT_WORLD ); }

    double time() const {
        double t;
        TSGetTime(ts, &t);
        return t;
    }
    
    unsigned step() const {
        int t;
        TSGetTimeStepNumber(ts, &t);
        return static_cast<unsigned>(t);
    }

    double dt() const {
        double dt;
        TSGetTimeStep(ts, &dt);
        return dt;
    }

    // void retain_stages(bool flg) {
    //     TSSetRetainStages(ts, flg ? PETSC_TRUE : PETSC_FALSE);
    // }

    Vector interpolate(double t) const {
        auto U = this->solution().duplicate();
        TSInterpolate(ts, t, U.v_);
        return U;
    }

    Vector solution() const {
        Vec U;
        TSGetSolution(ts, &U);
        return Vector(U, Vector::owner::other);
    }

    template <typename Mon>
    void set_monitor( Mon m )
    {
        static Mon m_ = m;
        TSMonitorSet( ts, TimeStepper::Monitor_function<Mon>, &m_, NULL );
    }

    template <typename Mon>
    static PetscErrorCode
    Monitor_function( TS ts_, int step, double t, Vec u, void* monitor )
    {
        TimeStepper T( ts_, false );
        Vector U( u, Vector::owner::other );
        ( *(Mon*)monitor )( T, step, t, U );
        return 0;
    }

    template <typename Jac>
    static PetscErrorCode
    RHSJacobian_function( TS ts, double t_, Vec u, Mat A, Mat B, void* G_u )
    {
        Vector U( u, Vector::owner::other );
        Matrix AA( A, false );
        Matrix BB( B, false );
        TimeStepper T( ts, false );
        ( *(Jac*)G_u )( U, AA, BB, T, t_ );
        return 0;
    }

    template <typename RHS>
    static PetscErrorCode
    RHS_function( TS ts, double t_, Vec u, Vec f, void* G )
    {
        Vector U( u, Vector::owner::other );
        Vector F( f, Vector::owner::other );
        TimeStepper T( ts, false );
        ( *(RHS*)G )( U, F, T, t_ );
        return 0;
    }

    TS ts;
    bool owned{true};
};
}
