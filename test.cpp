#include<thread>

#define SLEPC
#include"include/Petsc.hpp"


int main( int argc, const char** argv )
{
    petsc::PetscContext pc(argc, argv);
    {
        petsc::Matrix m(10);

        int rank;
        MPI_Comm_rank(m.comm(), &rank);

        m.reserve( [](int i, int j) { return i==j || i == j+1 || i == j+2 || i == j-1 || i == j-2; } );
        petsc::populate_matrix(m, [](int i, int j) {return i==j || i == j+1 || i == j+2 || i == j-1 || i == j-2; }, [](int i, int j){ return 1; });
        m.assemble(); m.print();

        auto t1 = std::thread(petsc::populate_matrix, 
                std::ref(m), 
                [](int i, int j) {return i==j; }, 
                [](int i, int j) { return i; }, 
                true);
        auto t2 = std::thread(petsc::populate_matrix, 
                std::ref(m), 
                [](int i, int j) {return i==j-1; }, 
                [](int i, int j) { return 2; }, 
                true);
        auto t3 = std::thread(petsc::populate_matrix, 
                std::ref(m), 
                [](int i, int j) {return i==j-2; }, 
                [](int i, int j) { return -2; }, 
                true);

        t1.join();
        t2.join();
        t3.join();
        m.assemble();
        m.print();

        auto vecs = m.get_vectors();

        petsc::populate_vector(vecs[0], [](int i) { return i; });
        vecs[0].assemble();
        vecs[0].print();
        std::cout << std::flush;

        //if (rank == 0) std::cout << "Eigentest!" << std::endl;
        petsc::EigenvalueSolver e(m, 10);

        e.print();

        //auto ev =  (*(e.begin())).evalue;
        //if (rank==0) std::cout << ev << std::endl;

        for( auto a: e)
        {
            if (rank==0) std::cerr << a.evalue << std::endl;
        }
    }
}
