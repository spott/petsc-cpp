#include <thread>
#include <functional>

#define SLEPC
#include "include/petsc_cpp/Petsc.hpp"


int main( int argc, const char** argv )
{
    using namespace petsc;
    PetscContext pc( argc, argv );
    Matrix m( 10 );

    int rank;
    MPI_Comm_rank( m.comm(), &rank );

    m.reserve( []( int i, int j ) {
        return i == j || i == j + 1 || i == j + 2 || i == j - 1 || i == j - 2;
    } );
    populate_matrix( m, []( int i, int j ) {
                            return i == j || i == j + 1 || i == j + 2 ||
                                   i == j - 1 || i == j - 2;
                        },
                     []( int, int ) { return 1; } );
    m.assemble();
    m.print();

    auto t1 = std::thread( petsc::populate_matrix, std::ref( m ),
                           []( int i, int j ) { return i == j; },
                           []( int i, int ) { return i; }, true );
    auto t2 = std::thread( petsc::populate_matrix, std::ref( m ),
                           []( int i, int j ) { return i == j - 1; },
                           []( int, int ) { return 2; }, true );
    auto t3 = std::thread( petsc::populate_matrix, std::ref( m ),
                           []( int i, int j ) { return i == j - 2; },
                           []( int, int ) { return -2; }, true );

    t1.join();
    t2.join();
    t3.join();
    m.assemble();
    m.print();

    auto vecs = m.get_vectors();

    populate_vector( vecs[0], []( int i ) { return i; } );
    vecs[0].assemble();
    vecs[0].print();

    std::cout << std::flush;

    vecs[1] = m * vecs[0];
    vecs[1].print();

    m.hermitian_transpose();
// if (rank == 0) std::cout << "Eigentest!" << std::endl;
#ifdef SLEPC
    EigenvalueSolver e( m, 10 );

    e.print();

    // auto ev =  (*(e.begin())).evalue;
    // if (rank==0) std::cout << ev << std::endl;

    for ( auto a : e ) {
        if ( rank == 0 ) std::cerr << a.evalue << std::endl;
    }
#endif


    // std::function<typename TimeStepper::jacobian_function_type> jac =
    // [](
    // TS ts, PetscReal t, Vec v, Mat * A, Mat * B, MatStructure * ms )
    //->PetscErrorCode
    //{

    // B = A;
    //// v.print();
    //// A.print();
    //// B.print();

    // std::cout << "this is the rhs function at time : " << t <<
    // std::endl;
    // std::cout << "and then I call the destructors on all these nice "
    //"objects that I created... :( " << std::endl;
    // return 0;
    //}
    //;

    // TimeStepper ts{vecs[0], 0, 1.0, .01, jac, m};

    std::cerr << std::endl << "this is the end!" << std::endl << std::endl;
}
