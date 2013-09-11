#pragma once

#include<memory>
#include<iostream>
#include<array>
#include<mutex>
#include<cassert>

namespace petsc
{

    //enum class ProblemType { Hermitian, GeneralizedHermitian, NonHermitian, GeneralizedNonHermitian, PositiveGeneralizedNonHermitian, GeneralizedHermitianIndefinite };
    //template<enum PT>
    //We are only going to look at non-generalized solvers initially, later we will template this with the above enum.
    class EigenvalueSolver
    {
        public:

            //EigenvalueSolver(const MPI_Comm& comm, Matrix& A, Matrix& B, int dim, EPSProblemType type, EPSWhich which_pair) {
                //EPSCreate(comm, &e_);
                //EPSSetOperators(e_, A.m_, B.m_);
                //EPSSetProblemType(e_, type);
                //EPSSetDimensions(e_, dim, PETSC_DECIDE , PETSC_DECIDE);
                //EPSSetWhichEigenpairs(e_, which_pair);
                //EPSSetFromOptions(e_);
                //EPSSolve(e_);
            //}

            EigenvalueSolver(const MPI_Comm& comm, Matrix& A, int dim, EPSWhich which_pair = EPS_SMALLEST_REAL, EPSProblemType type = EPS_HEP ): op_(A) {
                assert( type == EPS_HEP || type == EPS_NHEP ); //we only want hermitian or non-hermitian values
                EPSCreate(comm, &e_);
                EPSSetOperators(e_, A.m_, PETSC_NULL);
                EPSSetProblemType(e_, type);
                EPSSetDimensions(e_, dim, PETSC_DECIDE , PETSC_DECIDE);
                EPSSetWhichEigenpairs(e_, which_pair);
                EPSSetFromOptions(e_);
                EPSSolve(e_);
            }
            EigenvalueSolver(Matrix& A, int dim = 1): EigenvalueSolver(A.comm(), A, dim, EPS_SMALLEST_REAL, EPS_HEP) {}

            //rule of 4.5... 
            EigenvalueSolver( const EigenvalueSolver& other ) = delete; //we don't need no stinkin copy constructor...

            ~EigenvalueSolver() { EPSDestroy(&e_); }

            //Getters:
            MPI_Comm comm() {
                static MPI_Comm comm = [=]() {
                    MPI_Comm c;
                    PetscObjectGetComm((PetscObject)e_,&c);
                    return c;
                }();
                return comm;
            }

            //can't controll the order of the destruction of static objects...
            //so this object can't be static, because it won't be destroyed till it is too late.
            Matrix op() {
                return op_;
            }

            int iteration_number(){
                int its;
                EPSGetIterationNumber(e_,&its);
                return its;
            }

            std::array<int, 3> dimensions(){
                std::array<int, 3> out;
                EPSGetDimensions(e_,&(out[0]),&(out[1]),&(out[2]));
                return out;
            }

            std::tuple<PetscReal, PetscInt> tolerances() {
                PetscReal tol;
                PetscInt its;
                EPSGetTolerances(e_,&tol,&its);
                return std::tie(tol, its);
            }

            int num_converged() {
                static int n = [=](){ 
                    int nconv;
                    EPSGetConverged(e_,&nconv);
                    return nconv; } ();
                return n;
            }

            struct result {
                int nev;
                PetscScalar evalue;
                Vector evector;
            };

            //print!:
            void print() {
                EPSView(e_, PETSC_VIEWER_STDOUT_WORLD);
            }

            //This version spits out a new vector:
            EigenvalueSolver::result get_eigenpair(int nev) {
                auto v = op().get_vector();
                PetscScalar ev;
                EPSGetEigenpair(e_, nev, &ev, PETSC_NULL, v.v_, PETSC_NULL);
                return result{ nev, ev, v };
            }

            class iterator : public std::iterator< std::forward_iterator_tag, EigenvalueSolver::result, int>
            {
                public:
                    int nev;
                    EigenvalueSolver& e;

                    explicit iterator(const int i, EigenvalueSolver& es) : nev(i), e(es) {}

                    iterator( const iterator& i) : nev(i.nev), e(i.e) {};

                    iterator( iterator&& i) : nev(i.nev), e(i.e) {}

                    iterator& operator=( iterator rhs ) {
                        *this = iterator(rhs.nev, rhs.e);
                        return *this;
                    }

                    iterator& operator++() {
                        nev++;
                        return *this;
                    }
                    iterator operator++(int) {
                        auto ret = *this;
                        nev++;
                        return ret;
                    }

                    //iterator& operator+=(iterator::Distance diff) {
                        //nev+=diff;
                        //return *this;
                    //}
                    //iterator& operator--() {
                        //nev--;
                        //return *this;
                    //}
                    //iterator& operator--(int) {
                        //auto ret = *this;
                        //nev--;
                        //return ret;
                    //}

                    //iterator& operator-=(iterator::Distance diff) {
                        //nev-=diff;
                        //return *this;
                    //}


                    bool operator!=( const iterator& rhs ){
                        return (rhs.nev != nev || (&(rhs.e) != &(e)));
                    }

                    iterator::value_type operator*() {
                        return e.get_eigenpair(nev);
                    }

            };

            EigenvalueSolver::iterator begin() {
                return EigenvalueSolver::iterator(0, *this);
            }

            EigenvalueSolver::iterator end() {
                return EigenvalueSolver::iterator(num_converged()-1, *this);
            }

            Matrix& op_;
            EPS e_;
    };
}




