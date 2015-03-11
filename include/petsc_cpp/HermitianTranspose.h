#pragma once

typedef struct {
    Mat A;
} Mat_HTranspose;

PetscErrorCode MatMult_HTranspose( Mat N, Vec x, Vec y );

PetscErrorCode MatMultAdd_HTranspose( Mat N, Vec v1, Vec v2, Vec v3 );

PetscErrorCode MatMultHTranspose_HTranspose( Mat N, Vec x, Vec y );

PetscErrorCode MatMultHTransposeAdd_HTranspose( Mat N, Vec v1, Vec v2, Vec v3 );

PetscErrorCode MatDestroy_HTranspose( Mat N );

PetscErrorCode MatCreateHTranspose( Mat A, Mat* N );
