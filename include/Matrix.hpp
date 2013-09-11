#pragma once

#include<memory>
#include<iostream>
#include<array>
#include<mutex>

#include<petsc.h>

namespace petsc{

class Matrix {
    public:

        //constructors:
        Matrix(const MPI_Comm comm = PETSC_COMM_WORLD): has_type(false) {
            MatCreate(comm, &m_);
        }

        Matrix(const MatType t = MATMPIAIJ, const MPI_Comm comm = PETSC_COMM_WORLD) {
            MatCreate(comm, &m_);
            MatSetType(m_, t);
            has_type = true;
        }

        //symmetric
        Matrix(size_t n, const MatType t = MATMPIAIJ, const MPI_Comm comm = PETSC_COMM_WORLD) {
            MatCreate(comm, &m_);
            MatSetType(m_, t);
            has_type = true;
            MatSetSizes(m_, PETSC_DECIDE, PETSC_DECIDE, n, n);
        }

        //non-symmetric
        Matrix(size_t n, size_t m, const MatType t = MATMPIAIJ, const MPI_Comm comm = PETSC_COMM_WORLD) {
            MatCreate(comm, &m_);
            MatSetType(m_, t);
            has_type = true;
            MatSetSizes(m_, PETSC_DECIDE, PETSC_DECIDE, n, m);
        }

        // assume that the vector actually has a type... this should only be used in the implementation, but might be used if
        // I forgot a function;
        Matrix( Mat& in ) : m_(in), has_type(true), assembled(true) {};

        //rule of 4.5:
        Matrix(const Matrix& other) {
            MatConvert(other.m_, MATSAME, MAT_INITIAL_MATRIX, &m_);
        }

        Matrix(Matrix&& other): m_(other.m_), has_type(other.has_type) {
            other.m_ = PETSC_NULL;
        }

        //assignment operator:
        Matrix& operator=( Matrix other) {
            swap(*this, other);
            return *this;
        }

        //destructor
        ~Matrix() {
            MatDestroy(&m_);
        }

        //swap!
        friend void swap(Matrix& first, Matrix& second) // nothrow
        {
            using std::swap;

            swap(first.has_type, second.has_type); 
            swap(first.assembled, second.assembled); 
            swap(first.m_, second.m_);
        }

        //modifiers:
        void set_type(const MatType t) {
            l.lock();
            MatSetType(m_, t);
            has_type = true;
            l.unlock();
        }
        void set_option(const MatOption o, bool b){
            l.lock();
            MatSetOption(m_, o, b ? PETSC_TRUE : PETSC_FALSE);
            l.unlock();
        }
        void set_size(const int n_global, const int m_global, const int n_local, const int m_local){
            l.lock();
            MatSetSizes(m_, n_local, n_local, n_global, m_global);
            l.unlock();
        }

        //set value:
        void set_value(const int n, const int m, PetscScalar v)
        {
            l.lock();
            MatSetValues(m_, 1, &n, 1, &m, &v, INSERT_VALUES);
            assembled = false;
            l.unlock();
        }

        //reserve space:
        template<typename F>
            void reserve( F test )
            {
                l.lock();
                MatSetUp(m_);

                PetscInt rowstart, rowend, colstart, colend;
                MatGetOwnershipRange(m_, &rowstart, &rowend);
                MatGetOwnershipRangeColumn(m_, &colstart, &colend);
                PetscInt dnnz[rowend-rowstart];
                PetscInt onnz[rowend-rowstart];
                //find the preallocation functions:
                for (size_t i = rowstart; i < rowend; i++)
                {
                    dnnz[i-rowstart] = 0;
                    onnz[i-rowstart] = 0;
                    for (size_t j = 0; j < n()[0]; j++)
                    {
                        if (test(i,j))
                        {
                            if (j >= colstart && j < colend)
                                dnnz[i-rowstart]++;
                            else
                                onnz[i-rowstart]++;
                        }
                    }
                }
                MatMPIAIJSetPreallocation(m_, 0, dnnz, 0, onnz);
                l.unlock();
            }

        //assemble!
        void assemble() {
            l.lock();
            MatAssemblyBegin(m_,MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(m_,MAT_FINAL_ASSEMBLY);
            assembled = true;
            l.unlock();
        }

        //getters:
        MPI_Comm comm() {
            static MPI_Comm comm = [=]() {
                MPI_Comm c;
                PetscObjectGetComm((PetscObject)m_,&c);
                return c;
            }();
            return comm;
        }

        std::array<int,2> n() {
            static std::array<int,2> n = [=]() { 
                std::array<int,2> m; 
                MatGetSize(m_, &m[0], &m[1]);
                return m;
            }();
            return n;
        }

        std::array<int,2> get_ownership_rows() {
            static std::array<int,2> n = [=]() {
                std::array<int,2> m;
                MatGetOwnershipRange(m_, &m[0], &m[1]);
                return m; }();
            return n;
        }

        std::array<Vector, 2> get_vectors() {
            Vec a, b;
            MatGetVecs(m_,&a,&b);
            return std::array<Vector, 2>{ Vector{a},Vector{b} };
        }
        
        Vector get_vector() {
            Vec a;
            MatGetVecs(m_,&a, PETSC_NULL);
            return Vector{a};
        }

        //print!
        void print() {
            MatView(m_, PETSC_VIEWER_STDOUT_WORLD);
        }


        //this is public incase people want to use methods that aren't defined yet;
        Mat m_;
    private:

        //state:
        bool has_type;
        bool assembled;
        std::mutex l;

        friend class Vector;
};

}
