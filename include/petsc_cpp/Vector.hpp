#pragma once

#include <memory>
#include <iostream>
#include <array>
#include <mutex>
#include <vector>
#include <cassert>
#include <cstring>


namespace petsc
{
class Vector
{

  public:
    enum class type : char { seq, mpi, standard };

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
    Vector( size_t size,
            const type t = type::standard,
            MPI_Comm comm = PETSC_COMM_WORLD )
    {
        if ( t == type::seq ) comm = PETSC_COMM_SELF;
        VecCreate( comm, &v_ );
        VecSetType( v_, to_VecType( t ) );
        VecSetSizes( v_, PETSC_DECIDE, static_cast<int>( size ) );
    }

    Vector( const std::complex<double>* input,
            const size_t size,
            const type t = type::seq )
    {
        assert( t != type::standard );
        if ( t == type::seq )
            VecCreateSeqWithArray( PETSC_COMM_SELF, 1, static_cast<int>( size ),
                                   input, &v_ );
        if ( t == type::mpi )
            VecCreateMPIWithArray( PETSC_COMM_WORLD, 1,
                                   static_cast<int>( size ), PETSC_DECIDE,
                                   input, &v_ );
    }


    /*************
    //rule of 4.5:
    *************/
    // copy constructor:
    Vector( const Vector& other )
    {
        VecDuplicate( other.v_, &v_ );
        VecCopy( other.v_, v_ );
    }

    // move constructor:
    Vector( Vector&& other ) : v_( other.v_ ) { other.v_ = PETSC_NULL; }

    // this should only be used in the implementation, but might be used if I
    // forgot a function;
    enum class owner : char { self, other, bv };
    Vector( Vec& in, owner o = owner::self ) : v_( in )
    {
        switch ( o ) {
            case owner::self:
                deleter = []( Vec v ) { VecDestroy( &v ); };
                break;
            case owner::other:
                deleter = []( Vec ) {};
                break;
            case owner::bv:
                throw std::invalid_argument(
                    "bv can't be made with this constructor" );
        }
    }

    Vector( Vec& in, std::function<void( Vec )> deleter_ )
        : v_( in ), deleter( deleter_ )
    {
    }

    // assignment operator:
    Vector& operator=( Vector other )
    {
        swap( *this, other );
        return *this;
    }

    // destructor:
    ~Vector() { deleter( v_ ); }

    friend void swap( Vector& first, Vector& second ); // nothrow

    /*************
    //modifiers:
    *************/

    // set value:
    void set_value( const int n, PetscScalar v );

    Vector& set_all( const PetscScalar v );

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
    Vector& operator*=( const Vector& other );
    Vector& operator/=( const PetscScalar& other );
    Vector& axpy(const PetscScalar& alpha, const Vector& x);
    Vector& operator-=( const Vector& other );
    Vector& operator+=( const Vector& other );
    /*************
    //getters:
    *************/
    MPI_Comm comm() const;
    int rank() const;

    size_t size() const;

    std::array<int, 2> get_ownership_rows() const;

    Vector duplicate() const;
    void print() const;
    static void draw( const std::vector<Vector>& vectors,
                      const std::vector<double>* v = nullptr );
    void draw( const std::vector<double>* v = nullptr ) const;
    void draw( std::function<double(double)> f,
               const std::vector<double>* v = nullptr ) const;

    void to_file( const std::string& filename ) const;


    // this is public incase people want to use methods that aren't defined
    // yet;
    Vec v_;

    // TODO:  make iterator for vector?

  private:
    // state:
    std::function<void( Vec )> deleter{[]( Vec v ) { VecDestroy( &v ); }};
    friend class Matrix;
};

Vector operator*( const PetscScalar& alpha, Vector b );
Vector conjugate( Vector v );
Vector abs_square( Vector v );
}
