#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>

#include<petsc_cpp/Petsc.hpp>

namespace petsc
{
class Vector
{
  public:
    Vector( const MPI_Comm comm = PETSC_COMM_WORLD )
        : has_type( false ), assembled( false )
    {
        VecCreate( comm, &v_ );
    }

    Vector( int size = 10,
            const VecType t = VECSTANDARD,
            const MPI_Comm comm = PETSC_COMM_WORLD )
        : has_type( true ), assembled( false )
    {
        VecCreate( comm, &v_ );
        VecSetType( v_, t );
        VecSetSizes( v_, PETSC_DECIDE, size );
    }


    /*************
    //rule of 4.5:
    *************/
    // copy constructor:
    Vector( const Vector& other ) : has_type( other.has_type )
    {
        VecDuplicate( other.v_, &v_ );
        VecCopy( other.v_, v_ );
    }

    // move constructor:
    Vector( Vector&& other ) : v_( other.v_ ), has_type( other.has_type )
    {
        other.v_ = PETSC_NULL;
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


    /*************
    // calculations:
    *************/
    double norm( NormType nt = NORM_2 ) const;

    double normalize();

    /*************
    //getters:
    *************/
    MPI_Comm comm() const;

    size_t size() const;

    std::array<int, 2> get_ownership_rows() const;

    void print() const;

    void to_file( const std::string& filename ) const;

    // assume that the vector actually has a type... this should only be
    // used in
    // the implementation, but might be used if
    // I forgot a function;
    Vector( Vec& in ) : v_( in ), has_type( true ), assembled( true ) {}

    // this is public incase people want to use methods that aren't defined
    // yet;
    Vec v_;

  private:
    // state:
    bool has_type;
    bool assembled;
    std::mutex l;

    friend class Matrix;
};
}