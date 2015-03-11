#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <vector>
#include <cassert>
#include <string>


namespace petsc
{

class Matrix
{
  public:
    enum class type {
        aij,
        mpi_aij,
        seq_aij,
        block_aij,
        mpi_block_aij,
        seq_block_aij,
        symm_block_aij,
        mpi_symm_block_aij,
        seq_symm_block_aij,
        dense,
        mpi_dense,
        seq_dense,
        // shell
    };

  private:
    static bool block_type( type t )
    {
        return t == type::block_aij || t == type::seq_block_aij ||
               t == type::mpi_block_aij || t == type::symm_block_aij ||
               t == type::seq_symm_block_aij || t == type::mpi_symm_block_aij;
    }

    static MatType to_MatType( type t )
    {
        switch ( t ) {
            case type::aij:
                return MATAIJ;
            case type::seq_aij:
                return MATSEQAIJ;
            case type::mpi_aij:
                return MATMPIAIJ;

            case type::block_aij:
                return MATBAIJ;
            case type::seq_block_aij:
                return MATSEQBAIJ;
            case type::mpi_block_aij:
                return MATMPIAIJ;

            case type::symm_block_aij:
                return MATSBAIJ;
            case type::seq_symm_block_aij:
                return MATSEQSBAIJ;
            case type::mpi_symm_block_aij:
                return MATMPISBAIJ;

            case type::dense:
                return MATDENSE;
            case type::seq_dense:
                return MATSEQDENSE;
            case type::mpi_dense:
                return MATMPIDENSE;

                // case type::shell:
                // return MATSHELL;
        }
    }

    static type to_type( MatType t )
    {
        if ( std::strcmp( t, MATAIJ ) ) return type::aij;
        if ( std::strcmp( t, MATSEQAIJ ) ) return type::seq_aij;
        if ( std::strcmp( t, MATMPIAIJ ) ) return type::mpi_aij;

        if ( std::strcmp( t, MATBAIJ ) ) return type::block_aij;
        if ( std::strcmp( t, MATSEQBAIJ ) ) return type::seq_block_aij;
        if ( std::strcmp( t, MATMPIAIJ ) ) return type::mpi_block_aij;

        if ( std::strcmp( t, MATSBAIJ ) ) return type::symm_block_aij;
        if ( std::strcmp( t, MATSEQSBAIJ ) ) return type::seq_symm_block_aij;
        if ( std::strcmp( t, MATMPISBAIJ ) ) return type::mpi_symm_block_aij;

        if ( std::strcmp( t, MATDENSE ) ) return type::dense;
        if ( std::strcmp( t, MATSEQDENSE ) ) return type::seq_dense;
        if ( std::strcmp( t, MATMPIDENSE ) ) return type::mpi_dense;

        // if ( t == MATSHELL ) return type::shell;

        throw std::out_of_range( std::string( "type not supported " ) +
                                 static_cast<const char*>( t ) );
    }

  public:
    // constructors:
    Matrix( const MPI_Comm comm = PETSC_COMM_WORLD ) : has_type( false )
    {
        MatCreate( comm, &m_ );
    }

    Matrix( const type t /*= MATMPIAIJ*/,
            const MPI_Comm comm = PETSC_COMM_WORLD )
        : mat_type( t )
    {
        MatCreate( comm, &m_ );
        MatSetType( m_, to_MatType( t ) );
        has_type = true;
    }

    // non-square:
    Matrix( unsigned int N,
            unsigned int M,
            const type t = type::aij,
            const MPI_Comm comm = PETSC_COMM_WORLD )
        : Matrix( t, comm )
    {
        assert( N == M &&
                !( t == type::symm_block_aij || t == type::seq_symm_block_aij ||
                   t == type::mpi_symm_block_aij ) );
        MatSetSizes( m_, PETSC_DECIDE, PETSC_DECIDE, static_cast<int>( N ),
                     static_cast<int>( M ) );
    }

    // square:
    Matrix( unsigned int N,
            const type t = type::aij,
            const MPI_Comm comm = PETSC_COMM_WORLD )
        : Matrix( N, N, t, comm )
    {
    }

    // non-square, blocked:
    Matrix( unsigned int N,
            unsigned int M,
            unsigned int block_size,
            const type t = type::aij,
            const MPI_Comm comm = PETSC_COMM_WORLD )
        : Matrix( N, M, t, comm )
    {
        assert( block_type( mat_type ) );
        MatSetBlockSize( m_, block_size );
    }

    // assume that the matrix actually has a type... this should only be
    // used in the implementation, but might be used if I forgot a
    // function:
    Matrix( Mat in ) : m_( in )
    {
        MatType t;
        MatGetType( m_, &t );
        if ( t ) {
            mat_type = to_type( t );
            has_type = true;
        } else
            has_type = false;
        PetscBool b;
        MatAssembled( m_, &b );
        b == PETSC_TRUE ? assembled = true : assembled = false;
    }

    // rule of 4.5:
    Matrix( const Matrix& other )
    {
        MatType t;
        MatGetType( other.m_, &t );
        if ( t ) {
            mat_type = to_type( t );
            has_type = true;
        } else
            has_type = false;
        PetscBool b;
        MatAssembled( other.m_, &b );
        b == PETSC_TRUE ? assembled = true : assembled = false;
        MatConvert( other.m_, MATSAME, MAT_INITIAL_MATRIX, &m_ );
    }

    Matrix( Matrix&& other )
        : m_( other.m_ ), has_type( other.has_type ),
          assembled( other.assembled ), l(), mat_type( other.mat_type )
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
                   const int m_local,
                   const int n_block = 1,
                   const int m_block = 1 );

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
    int rank() const;

    Matrix& transpose();
    Matrix& hermitian_transpose();
    std::array<int, 2> n() const;

    std::array<int, 2> get_ownership_rows() const;

    std::array<Vector, 2> get_vectors() const;

    Vector get_right_vector() const;

    Vector get_left_vector() const;

    double norm( NormType t = NORM_FROBENIUS );

    Vector operator*( const Vector& v ) const;

    // print!
    void print() const;

    void to_file( const std::string& filename ) const;

    // this is public incase people want to use methods that aren't defined
    // yet:
    Mat m_;

  private:
    // state:
    bool has_type{false};
    bool assembled{false};
    std::mutex l;
    type mat_type;
    friend class Vector;
};

// return matrices that act as "transposes" of the given matrix
Matrix transpose( const Matrix& A );
Matrix hermitian_transpose( const Matrix& A );
}
