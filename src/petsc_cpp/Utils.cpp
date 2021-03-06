#include <petsc_cpp/Petsc.hpp>

namespace petsc
{


PetscScalar inner_product( const Vector& l, const Vector& r )
{
    PetscScalar out;
    VecDot( r.v_, l.v_, &out );
    return out;
}

PetscScalar inner_product( const Vector& l, const Vector& m, const Vector& r )
{
    PetscScalar out;
    Vector rm = m.duplicate();
    VecPointwiseMult( rm.v_, m.v_, r.v_ );
    VecDot( rm.v_, l.v_, &out );
    return out;
}

PetscScalar inner_product( const Vector& l, const Matrix& m, const Vector& r )
{
    PetscScalar out;
    Vector rm = m.get_left_vector();
    MatMult( m.m_, r.v_, rm.v_ );
    VecDot( rm.v_, l.v_, &out );
    return out;
}
}
