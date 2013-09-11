#pragma once

#include<memory>
#include<iostream>
#include<array>
#include<mutex>

namespace petsc {

    void populate_matrix( Matrix &m_, std::function< bool (int, int) > test, std::function< PetscScalar (int, int) > find_value, bool symmetric = true)
    {
        //Local objects:
        PetscInt rowstart, rowend;
        //PetscInt colstart, colend;

        int rank;
        MPI_Comm_rank(m_.comm(), &rank);

        auto ranges = m_.get_ownership_rows();
        rowstart = ranges[0];
        rowend = ranges[1];

        PetscScalar value;

        for (PetscInt i = rowstart; i < rowend; i++)
        {
            for (PetscInt j = (symmetric ? i : 0u); j < m_.n()[0]; j++)
            {
                if (test(i,j))
                {
                    value = find_value(i,j);
                    m_.set_value(i, j, value);
                    if (symmetric)
                        m_.set_value(j, i, value);
                }
            }
        }

    }

    void populate_vector( Vector &v_, std::function< PetscScalar (int) > values )
    {
        int rank;
        MPI_Comm_rank(v_.comm(), &rank);
        auto ranges = v_.get_ownership_rows();

        PetscScalar value;
        for (PetscInt i = ranges[0]; i < ranges[1]; ++i)
            v_.set_value(i, values(i));

    }

}
