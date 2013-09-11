#pragma once

#include<memory>
#include<iostream>
#include<array>
#include<mutex>

//#include<petsc.h>


namespace petsc
{
class Vector {
    public:
        Vector(const MPI_Comm comm = PETSC_COMM_WORLD): has_type(false), assembled(false) {
            VecCreate(comm, &v_);
        }

        Vector(int size = 10, const VecType t = VECSTANDARD, const MPI_Comm comm = PETSC_COMM_WORLD): has_type(true), assembled(false){
            VecCreate(comm, &v_);
            VecSetType(v_, t);
            VecSetSizes(v_, PETSC_DECIDE, size);
        }


        /*************
        //rule of 4.5:
        *************/
        //copy constructor:
        Vector( const Vector& other) : has_type(other.has_type) {
            VecDuplicate(other.v_, &v_);
            VecCopy(other.v_, v_);
        }

        //move constructor:
        Vector( Vector&& other ) : v_(other.v_), has_type(other.has_type) {
            other.v_ = PETSC_NULL;
        }

        //assignment operator:
        Vector& operator=( Vector other) {
            swap(*this, other);
            return *this;
        }

        //destructor:
        ~Vector() {
            VecDestroy(&v_);
        }

        friend void swap(Vector& first, Vector& second) // nothrow
        {
            using std::swap;

            swap(first.has_type, second.has_type); 
            swap(first.v_, second.v_);
        }

        /*************
        //modifiers:
        *************/

        //set value:
        void set_value( const int n, PetscScalar v)
        {
            l.lock();
            VecSetValue(v_, n, v, INSERT_VALUES);
            assembled = false;
            l.unlock();
        }

        //assemble:
        void assemble() {
            l.lock();
            VecAssemblyBegin(v_);
            VecAssemblyEnd(v_);
            assembled = true;
            l.unlock();
        }

        /*************
        //getters:
        *************/
        MPI_Comm comm() {
            static MPI_Comm comm = [=]() {
                MPI_Comm c;
                PetscObjectGetComm((PetscObject)v_,&c);
                return c;
            }();
            return comm;
        }

        size_t size() {
            static int n = [=](){
                int i;
                VecGetSize(v_, &i);
                return i; }();
            return static_cast<size_t>(n);
        }

        std::array<int,2> get_ownership_rows() {
            static std::array<int,2> n = [=]() {
                std::array<int,2> m;
                VecGetOwnershipRange(v_, &m[0], &m[1]);
                return m; }();
            return n;
        }

        void print() {
            VecView(v_, PETSC_VIEWER_STDOUT_WORLD);
        }

        // assume that the vector actually has a type... this should only be used in the implementation, but might be used if
        // I forgot a function;
        Vector(Vec& in) : v_(in), has_type(true), assembled(true) {};

        //this is public incase people want to use methods that aren't defined yet;
        Vec v_;

    private:

        //state:
        bool has_type;
        bool assembled;
        std::mutex l;

        friend class Matrix;
};

}
