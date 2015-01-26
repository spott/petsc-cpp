#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <vector>
#include <cassert>

#include <petsc_cpp/Petsc.hpp>

namespace petsc
{

class Matrix
{
  public:
    // constructors:
    Matrix( const MPI_Comm comm = PETSC_COMM_WORLD ) : has_type( false )
    {
        MatCreate( comm, &m_ );
    }

    Matrix( const MatType t = MATMPIAIJ,
            const MPI_Comm comm = PETSC_COMM_WORLD )
    {
        MatCreate( comm, &m_ );
        MatSetType( m_, t );
        has_type = true;
    }

    // symmetric:
    Matrix( unsigned int n,
            const MatType t = MATMPIAIJ,
            const MPI_Comm comm = PETSC_COMM_WORLD )
    {
        MatCreate( comm, &m_ );
        MatSetType( m_, t );
        has_type = true;
        MatSetSizes( m_, PETSC_DECIDE, PETSC_DECIDE, static_cast<int>( n ),
                     static_cast<int>( n ) );
    }

    // non-symmetric:
    Matrix( unsigned int n,
            unsigned int m,
            const MatType t = MATMPIAIJ,
            const MPI_Comm comm = PETSC_COMM_WORLD )
    {
        MatCreate( comm, &m_ );
        MatSetType( m_, t );
        has_type = true;
        MatSetSizes( m_, PETSC_DECIDE, PETSC_DECIDE, static_cast<int>( n ),
                     static_cast<int>( m ) );
    }

    // assume that the matrix actually has a type... this should only be
    // used in the implementation, but might be used if I forgot a
    // function:
    Matrix( Mat& in ) : m_( in ), has_type( true ), assembled( true ) {}

    // rule of 4.5:
    Matrix( const Matrix& other )
    {
        MatConvert( other.m_, MATSAME, MAT_INITIAL_MATRIX, &m_ );
    }

    Matrix( Matrix&& other ) : m_( other.m_ ), has_type( other.has_type )
    {
        other.m_ = PETSC_NULL;
    }

    // assignment operator:
    Matrix& operator=( Matrix other );

    // destructor:
    ~Matrix() { MatDestroy( &m_ ); }

    // swap!
    friend void swap( Matrix& first, Matrix& second ); // nothrow

    // modifiers:
    void set_type( const MatType t );

    void set_option( const MatOption o, bool b );

    void set_size( const int n_global,
                   const int m_global,
                   const int n_local,
                   const int m_local );

    // set value:
    void set_value( const int n, const int m, PetscScalar v );

    // reserve space:
    template <typename F>
    void reserve( F test )
    {
        l.lock();
        MatSetUp( m_ );

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsign-conversion"
        PetscInt rowstart, rowend, colstart, colend;
        MatGetOwnershipRange( m_, &rowstart, &rowend );
        MatGetOwnershipRangeColumn( m_, &colstart, &colend );
        assert( rowend >= rowstart );
        std::vector<int> dnnz( rowend - rowstart );
        std::vector<int> onnz( rowend - rowstart );
        // PetscInt dnnz[rowend - rowstart];
        // PetscInt onnz[rowend - rowstart];
        // find the preallocation functions:
        for ( auto i = rowstart; i < rowend; i++ ) {
            dnnz[i - rowstart] = 0;
            onnz[i - rowstart] = 0;
            for ( auto j = 0; j < n()[0]; j++ ) {
                if ( test( i, j ) ) {
                    if ( j >= colstart && j < colend )
                        dnnz[i - rowstart]++;
                    else
                        onnz[i - rowstart]++;
                }
            }
        }
#pragma clang diagnostic pop
        MatMPIAIJSetPreallocation( m_, 0, dnnz.data(), 0, onnz.data() );
        l.unlock();
    }

    // assemble!
    void assemble();

    // getters:
    MPI_Comm comm() const;

    std::array<int, 2> n() const;

    std::array<int, 2> get_ownership_rows() const;

    std::array<Vector, 2> get_vectors() const;

    Vector get_right_vector() const;

    Vector get_left_vector() const;

    Vector operator*( const Vector& v ) const;

    // print!
    void print() const;

    void to_file( const std::string& filename ) const;

    // this is public incase people want to use methods that aren't defined
    // yet:
    Mat m_;

  private:
    // state:
    bool has_type;
    bool assembled;
    std::mutex l;

    friend class Vector;
};
}
