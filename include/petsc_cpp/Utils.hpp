#pragma once

// c stdlib
#include <unistd.h>

// stl
#include <fstream>
#include <cstdarg>
#include <memory>
#include <iostream>
#include <array>
#include <mutex>

// petsc:

namespace petsc
{

inline void populate_matrix( Matrix& m_,
                             std::function<bool(int, int)> test,
                             std::function<PetscScalar(int, int)> find_value,
                             bool symmetric = true )
{
    // Local objects:
    PetscInt rowstart, rowend;
    // PetscInt colstart, colend;

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
    auto ranges = v_.get_ownership_rows();

    PetscScalar value;
    for ( PetscInt i = ranges[0]; i < ranges[1]; ++i )
        v_.set_value( i, values( i ) );
}

template <typename T>
inline void binary_import( T&, const std::string& );

template <>
inline void binary_import<Matrix>( Matrix& m_, const std::string& filename )
{
    PetscViewer view;
    PetscViewerBinaryOpen( m_.comm(), filename.c_str(), FILE_MODE_READ, &view );
    MatLoad( m_.m_, view );
    PetscViewerDestroy( &view );
}

template <>
inline void binary_import<Vector>( Vector& m_, const std::string& filename )
{
    PetscViewer view;
    PetscViewerBinaryOpen( m_.comm(), filename.c_str(), FILE_MODE_READ, &view );
    VecLoad( m_.v_, view );
    PetscViewerDestroy( &view );
}


template <typename T>
inline T& map( const T&, std::function<PetscScalar(PetscScalar, int)>, T& );

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
}

template <typename T>
inline T& map( T&, std::function<PetscScalar(PetscScalar, int)> );

template <>
inline Vector& map<Vector>( Vector& v,
                            std::function<PetscScalar(PetscScalar, int)> f )
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

PetscScalar inner_product( const Vector& l, const Matrix& m, const Vector& r );

// for diagonal vectors:
PetscScalar inner_product( const Vector& l, const Vector& m, const Vector& r );
}


namespace util
{

inline void wait_for_key()
{
    std::cout << std::endl << "Press ENTER to continue..." << std::endl;
    std::cin.clear();
    std::cin.ignore( std::cin.rdbuf()->in_avail() );
    std::cin.get();
}


inline void printProgBar( double percent, std::ostream& os = std::cout )
{
    percent *= 100;
    std::string bar;

    for ( size_t i = 0; i < 50; i++ ) {
        if ( i < ( percent / 2 ) ) {
            bar.replace( i, 1, "=" );
        } else if ( i == ( size_t( percent ) / 2 ) ) {
            bar.replace( i, 1, ">" );
        } else {
            bar.replace( i, 1, " " );
        }
    }

    os << "\r"
          "[" << bar << "] ";
    os.width( 3 );
    os << percent << "%     " << std::flush;
}


inline std::string absolute_path( const std::string& rel_path )
{
    if ( rel_path[0] == '.' ) {
        char* a = new char[1025];
        getcwd( a, 1025 );
        std::string cwd = std::string( a );
        delete a;
        return cwd.append( "/" ).append( rel_path );
    } else
        return rel_path;
}


inline bool file_exists( const std::string& fname )
{
    bool ret;
    std::ifstream f( fname );
    if ( f.good() )
        ret = true;
    else
        ret = false;
    f.close();

    return ret;
}

template <typename T, typename U>
inline void export_vector_binary( const std::string& filename,
                                  const std::vector<T>& out,
                                  const std::vector<U>& prefix )
{
    std::ios::pos_type size;
    std::ofstream file;
    file.open( filename.c_str(), std::ios::binary | std::ios::out );
    if ( file.is_open() ) {
        if ( prefix.size() > 0 )
            file.write( reinterpret_cast<const char*>( prefix.data() ),
                        static_cast<size_t>( sizeof( U ) * prefix.size() ) );
        file.write( reinterpret_cast<const char*>( &out[0] ),
                    static_cast<size_t>( sizeof( T ) * out.size() ) );
        file.close();
    } else {
        std::cerr << "error opening file... does the folder exist?: "
                  << filename << std::endl;
        throw new std::exception();
    }
}

template <typename T, typename T2 = T, size_t block_size = 100>
inline void export_vector_binary( const std::string& filename,
                                  const std::vector<T>& out,
                                  bool append = false )
{
    std::ios::pos_type size;
    std::ofstream file;
    auto openmode = std::ios::binary | std::ios::out;
    if ( append ) openmode = openmode | std::ios::app;
    file.open( filename.c_str(), openmode );
    if ( file.is_open() ) {
        for ( auto i = out.begin(); i < out.end(); i += block_size ) {
            std::array<T2, block_size> ni;
            for ( auto j = i; j < ( ( out.end() - i < int( block_size ) )
                                        ? out.end()
                                        : i + block_size );
                  j++ )
                ni[static_cast<size_t>( j - i )] = static_cast<T2>( *j );
            file.write(
                reinterpret_cast<const char*>( &ni ),
                static_cast<std::streamsize>(
                    sizeof( T2 ) * ( ( out.end() - i < int( block_size ) )
                                         ? static_cast<size_t>( out.end() - i )
                                         : block_size ) ) );
        }
        file.close();
    } else {
        std::cerr << "error opening file... does the folder exist?: "
                  << filename << std::endl;
        throw new std::exception();
    }
}
}

namespace functional
{

template <typename T>
inline void to_void( T& )
{
    return;
}

template <typename T>
inline T id( T& a )
{
    return a;
}

template <typename Scalar>
inline Scalar from_complex( std::complex<double> a );

template <>
std::complex<double> inline from_complex<std::complex<double>>(
    std::complex<double> a )
{
    return a;
}

template <>
inline double from_complex<double>( std::complex<double> a )
{
    return a.real();
}
}

namespace math
{
template <typename T>
inline constexpr int signum( T x, std::false_type /*is_signed*/ )
{
    return T( 0 ) < x;
}

template <typename T>
inline constexpr int signum( T x, std::true_type /*is_signed*/ )
{
    return ( T( 0 ) < x ) - ( x < T( 0 ) );
}

template <typename T>
inline constexpr int signum( T x )
{
    return signum( x, std::is_signed<T>() );
}
}
