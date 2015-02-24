#include <petsc-private/matimpl.h>          /*I "petscmat.h" I*/

typedef struct {
  Mat A;
} Mat_HTranspose;

#undef __FUNCT__
#define __FUNCT__ "MatMult_HTranspose"
PetscErrorCode MatMult_HTranspose(Mat N,Vec x,Vec y)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultHermitianTranspose(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultAdd_HTranspose"
PetscErrorCode MatMultAdd_HTranspose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultHermitianTransposeAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTranspose_HTranspose"
PetscErrorCode MatMultTranspose_HTranspose(Mat N,Vec x,Vec y)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMult(Na->A,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatMultTransposeAdd_HTranspose"
PetscErrorCode MatMultTransposeAdd_HTranspose(Mat N,Vec v1,Vec v2,Vec v3)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatMultAdd(Na->A,v1,v2,v3);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDestroy_HTranspose"
PetscErrorCode MatDestroy_HTranspose(Mat N)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&Na->A);CHKERRQ(ierr);
  ierr = PetscFree(N->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatDuplicate_HTranspose"
PetscErrorCode MatDuplicate_HTranspose(Mat N, MatDuplicateOption op, Mat* m)
{
  Mat_HTranspose  *Na = (Mat_HTranspose*)N->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  //at the moment, this doesn't NOT copy values
  if (op == MAT_COPY_VALUES)
    ierr = MatHermitianTranspose(N, MAT_INITIAL_MATRIX, m);
  else
    SETERRQ(PetscObjectComm((PetscObject)N),PETSC_ERR_ARG_SIZ,"Can't duplicate and not copy values");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatCreateHTranspose"
/*@
      MatCreateHTranspose - Creates a new matrix object that behaves like A'*

   Collective on Mat

   Input Parameter:
.   A  - the (possibly rectangular) matrix

   Output Parameter:
.   N - the matrix that represents A'*

   Level: intermediate

   Notes: The transpose A' is NOT actually formed! Rather the new matrix
          object performs the matrix-vector product by using the MatMultHermitianTranspose() on
          the original matrix

.seealso: MatCreateNormal(), MatMult(), MatMultHermitianTranspose(), MatCreate()

@*/
PetscErrorCode  MatCreateHTranspose(Mat A,Mat *N)
{
  PetscErrorCode ierr;
  PetscInt       m,n;
  Mat_HTranspose  *Na;

  PetscFunctionBegin;
  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatCreate(PetscObjectComm((PetscObject)A),N);CHKERRQ(ierr);
  ierr = MatSetSizes(*N,n,m,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->rmap);CHKERRQ(ierr);
  ierr = PetscLayoutSetUp((*N)->cmap);CHKERRQ(ierr);
  //hopefully "MATTRANSPOSEMAT" is good enough
  ierr = PetscObjectChangeTypeName((PetscObject)*N,MATTRANSPOSEMAT);CHKERRQ(ierr);

  ierr       = PetscNewLog(*N,&Na);CHKERRQ(ierr);
  (*N)->data = (void*) Na;
  ierr       = PetscObjectReference((PetscObject)A);CHKERRQ(ierr);
  Na->A      = A;

  (*N)->ops->destroy          = MatDestroy_HTranspose;
  (*N)->ops->mult             = MatMult_HTranspose;
  (*N)->ops->multadd          = MatMultAdd_HTranspose;
  (*N)->ops->multtranspose    = MatMultTranspose_HTranspose;
  (*N)->ops->multtransposeadd = MatMultTransposeAdd_HTranspose;
  (*N)->ops->duplicate        = MatDuplicate_HTranspose;
  (*N)->assembled             = PETSC_TRUE;

  ierr = MatSetBlockSizes(*N,PetscAbs(A->cmap->bs),PetscAbs(A->rmap->bs));CHKERRQ(ierr);
  ierr = MatSetUp(*N);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
