#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>

namespace petsc
{

inline void
populate_matrix( Matrix& m_,
                 std::function<bool(int, int)> test,
                 std::function<PetscScalar(int, int)> find_value,
                 bool symmetric = true )
{
    // Local objects:
    PetscInt rowstart, rowend;
    // PetscInt colstart, colend;

    int rank;
    MPI_Comm_rank( m_.comm(), &rank );

    auto ranges = m_.get_ownership_rows();
    rowstart = ranges[0];
    rowend = ranges[1];

    PetscScalar value;

    for ( PetscInt i = rowstart; i < rowend; i++ ) {
        for ( PetscInt j = ( symmetric ? i : 0u ); j < m_.n()[0]; j++ ) {
            if ( test( i, j ) ) {
                value = find_value( i, j );
                m_.set_value( i, j, value );
                if ( symmetric ) m_.set_value( j, i, value );
            }
        }
    }
}

inline void populate_vector( Vector& v_,
                             std::function<PetscScalar(int)> values )
{
    int rank;
    MPI_Comm_rank( v_.comm(), &rank );
    auto ranges = v_.get_ownership_rows();

    PetscScalar value;
    for ( PetscInt i = ranges[0]; i < ranges[1]; ++i )
        v_.set_value( i, values( i ) );
}

template <typename T>
inline void binary_import( T&, const std::string& );

template <>
inline void binary_import<Matrix>( Matrix& m_,
                                   const std::string& filename )
{
    PetscViewer view;
    PetscViewerBinaryOpen( m_.comm(), filename.c_str(), FILE_MODE_READ,
                           &view );
    MatLoad( m_.m_, view );
    PetscViewerDestroy( &view );
}

template <>
inline void binary_import<Vector>( Vector& m_,
                                   const std::string& filename )
{
    PetscViewer view;
    PetscViewerBinaryOpen( m_.comm(), filename.c_str(), FILE_MODE_READ,
                           &view );
    VecLoad( m_.v_, view );
    PetscViewerDestroy( &view );
}


template <typename T>
inline T&
map( const T&, std::function<PetscScalar(PetscScalar, int)>, T& );

template <>
inline Vector& map<Vector>( const Vector& v_,
                            std::function<PetscScalar(PetscScalar, int)> f,
                            Vector& out )
{
    int vstart, vend;
    int ostart, oend;
    PetscScalar const* a;
    PetscScalar* b;

    VecGetOwnershipRange( v_.v_, &vstart, &vend );
    VecGetOwnershipRange( out.v_, &ostart, &oend );
    VecGetArrayRead( v_.v_, &a );
    VecGetArray( out.v_, &b );

    assert( ( ostart == vstart ) && ( oend == vend ) );

    for ( int i = 0; i < oend - ostart; ++i ) b[i] = f( a[i], i + ostart );

    VecRestoreArrayRead( v_.v_, &a );
    VecRestoreArray( out.v_, &b );

    return out;
    ;
}

template <typename T>
inline T& map( T&, std::function<PetscScalar(PetscScalar, int)> );

template <>
inline Vector&
map<Vector>( Vector& v, std::function<PetscScalar(PetscScalar, int)> f )
{
    int vstart, vend;
    PetscScalar* a;

    VecGetOwnershipRange( v.v_, &vstart, &vend );
    VecGetArray( v.v_, &a );

    for ( int i = 0; i < vend - vstart; ++i ) a[i] = f( a[i], i + vstart );

    VecRestoreArray( v.v_, &a );

    return v;
}

PetscScalar inner_product( const Vector& l, const Vector& r );

PetscScalar
inner_product( const Vector& l, const Matrix& m, const Vector& r );
}
