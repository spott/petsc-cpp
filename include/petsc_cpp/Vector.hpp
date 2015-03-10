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
class Vector
{

  public:
    enum class type {
        seq,
        mpi,
        standard
        //#define VECSEQ         "seq"
        //#define VECMPI         "mpi"
        //#define VECSTANDARD    "standard"
    };

  private:
    static VecType to_VecType( type t )
    {
        switch ( t ) {
            case type::seq:
                return VECSEQ;
            case type::mpi:
                return VECMPI;
            case type::standard:
                return VECSTANDARD;
        }
    }

    static type to_type( VecType t )
    {
        if ( std::strcmp( t, VECSEQ ) ) return type::seq;
        if ( std::strcmp( t, VECMPI ) ) return type::mpi;
        if ( std::strcmp( t, VECSTANDARD ) ) return type::standard;
        throw std::out_of_range( std::string( "type not supported |" ) +
                                 static_cast<const char*>( t ) + "|" );
    }


  public:
    Vector( const MPI_Comm comm = PETSC_COMM_WORLD )
        : has_type( false ), assembled( false )
    {
        VecCreate( comm, &v_ );
    }

    Vector( int size,
            const type t = type::standard,
            MPI_Comm comm = PETSC_COMM_WORLD )
        : has_type( true ), assembled( false ), vec_type( t )
    {
        if ( t == type::seq ) comm = PETSC_COMM_SELF;
        VecCreate( comm, &v_ );
        VecSetType( v_, to_VecType( t ) );
        VecSetSizes( v_, PETSC_DECIDE, size );
    }


    /*************
    //rule of 4.5:
    *************/
    // copy constructor:
    Vector( const Vector& other )
        : has_type( other.has_type ), vec_type( other.vec_type )
    {
        VecDuplicate( other.v_, &v_ );
        VecCopy( other.v_, v_ );
    }

    // move constructor:
    Vector( Vector&& other )
        : v_( other.v_ ), has_type( other.has_type ),
          assembled( other.assembled ), vec_type( other.vec_type ), l()
    {
        other.v_ = PETSC_NULL;
    }

    // assume that the vector actually has a type... this should only be
    // used in
    // the implementation, but might be used if
    // I forgot a function;
    Vector( Vec& in ) : v_( in ), assembled( true )
    {
        VecType t;
        VecGetType( v_, &t );
        if ( t ) {
            vec_type = to_type( t );
            has_type = true;
        } else
            has_type = false;
    }

    // assignment operator:
    Vector& operator=( Vector other )
    {
        swap( *this, other );
        return *this;
    }

    // destructor:
    ~Vector() { VecDestroy( &v_ ); }

    friend void swap( Vector& first, Vector& second ); // nothrow

    /*************
    //modifiers:
    *************/

    // set value:
    void set_value( const int n, PetscScalar v );

    // assemble:
    void assemble();

    Vector& conjugate();

    /*************
    // calculations:
    *************/
    double norm( NormType nt = NORM_2 ) const;
    double normalize();
    std::complex<double> normalize_sign();

    Vector operator-( Vector other ) const;
    Vector operator+( Vector other ) const;
    Vector operator*( const PetscScalar& other ) const;
    Vector operator/( const PetscScalar& other ) const;
    Vector& operator*=( const PetscScalar& other );
    Vector& operator/=( const PetscScalar& other );

    /*************
    //getters:
    *************/
    MPI_Comm comm() const;
    int rank() const;

    size_t size() const;

    std::array<int, 2> get_ownership_rows() const;

    Vector duplicate() const;
    void print() const;
    void draw( const std::vector<double>* v = nullptr ) const;

    void to_file( const std::string& filename ) const;


    // this is public incase people want to use methods that aren't defined
    // yet;
    Vec v_;

    // TODO:  make iterator for vector?

  private:
    // state:
    bool has_type;
    bool assembled;
    type vec_type;
    std::mutex l;

    friend class Matrix;
};

Vector operator*( const PetscScalar& alpha, Vector b );
Vector conjugate( Vector v );
}
