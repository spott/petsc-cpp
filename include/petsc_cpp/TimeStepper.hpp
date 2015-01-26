#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <cassert>

namespace petsc
{
class TimeStepper;
namespace TimeStepper_private
{
    // template< typename context >
    std::function<PetscErrorCode(TS, PetscReal, Vec, Mat*, Mat*, MatStructure*)>
    jacobian;

    PetscErrorCode jacobian_function_( TS ts,
                                       PetscReal t,
                                       Vec u,
                                       Mat* A,
                                       Mat* B,
                                       MatStructure* ms,
                                       void* cntx );

    // template< typename context >
    std::function<PetscErrorCode( TS, PetscReal, Vec, Vec )> rhs;

    PetscErrorCode
    rhs_function_( TS ts, PetscReal t, Vec v, Vec f, void* cntx );
}
class TimeStepper
{
  public:

    typedef PetscErrorCode( jacobian_function_type )(
        TS, PetscReal, Vec, Mat*, Mat*, MatStructure* );

    typedef PetscErrorCode( rhs_function_type )( TS, PetscReal, Vec, Vec );
    // This won't immediately run, because that is impractical (a lot of
    // optional setup)
    TimeStepper( const MPI_Comm& comm,
                 TSProblemType pt,
                 TSType type,
                 PetscReal t0,
                 PetscReal dt,
                 Vector& solution,
                 PetscReal tf,
                 PetscInt max_step,
                 std::function<jacobian_function_type> jac,
                 std::function<rhs_function_type> rhs,
                 Matrix& rhs_mat_initial )
    {
        TSCreate( comm, &ts_ );
        TSSetProblemType( ts_, pt );
        TSSetType( ts_, type );
        TSSetInitialTimeStep( ts_, t0, dt );
        TSSetSolution( ts_, solution.v_ );
        if ( max_step > 0 )
            TSSetDuration( ts_, static_cast<int>( ( tf - t0 ) / dt ), tf );
        else
            TSSetDuration( ts_, max_step, tf );
        TSSetFromOptions( ts_ );

        TSSetRHSJacobian( ts_,
                          rhs_mat_initial.m_,
                          rhs_mat_initial.m_,
                          TimeStepper_private::jacobian_function_,
                          NULL );

        TimeStepper_private::jacobian = jac;
        if ( pt == TS_LINEAR )
            TSSetRHSFunction( ts_, NULL, TSComputeRHSFunctionLinear, NULL );
        else {
            TSSetRHSFunction(
                ts_, NULL, TimeStepper_private::rhs_function_, NULL );
            TimeStepper_private::rhs = rhs;
        }


        TSSolve( ts_, PETSC_NULL );
    }

    explicit TimeStepper( TS& ts ) : ts_( ts ) {}

    TimeStepper( Vector& solution,
                 PetscReal t0,
                 PetscReal tf,
                 PetscReal dt,
                 std::function<jacobian_function_type> jac,
                 Matrix& rhs_mat_initial,
                 TSProblemType pt = TS_LINEAR,
                 TSType type = TSCN )
        : TimeStepper( solution.comm(),
                       pt,
                       type,
                       t0,
                       dt,
                       solution,
                       tf,
                       -1,
                       jac,
                       nullptr,
                       rhs_mat_initial ) {}

    TimeStepper( const TimeStepper& other ) = delete;

    TimeStepper( TimeStepper&& other ) : ts_( other.ts_ )
    {
        other.ts_ = PETSC_NULL;
    }

    ~TimeStepper()
    {
        TSDestroy( &ts_ );
    }

    // func (TS ts,PetscReal t,Vec u,Vec F,void *ctx);
    // void rhs_function( std::function<PetscErrorCode(
    // TimeStepper, PetscReal, Vector, Vector )> rhs_function_ )
    //{
    // TimeStepper_private::rhs = rhs_function_;
    //}

    //// func (TS ts,PetscReal t,Vec u,Mat *A,Mat *B,MatStructure *flag,void
    //// *ctx);
    // void rhs_jacobian( std::function<PetscErrorCode(
    // TS, PetscReal, Vec, Mat*, Mat*, MatStructure*)>
    // jac_function_ )
    //{
    // TimeStepper_private::jacobian = jac_function_;
    //}

    TS ts_;
};


PetscErrorCode TimeStepper_private::jacobian_function_(
    TS ts, PetscReal t, Vec u, Mat* A, Mat* B, MatStructure* ms, void* cntx )
{
    return TimeStepper_private::jacobian( ts, t, u, A, B, ms );
}
PetscErrorCode TimeStepper_private::rhs_function_(
    TS ts, PetscReal t, Vec v, Vec f, void* cntx )
{
    return TimeStepper_private::rhs( ts, t, v, f );
}
}
