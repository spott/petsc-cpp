PETSC_DIR=/usr/local/Cellar/petsc/3.5.2-debug
SLEPC_DIR=/usr/local/Cellar/slepc/3.5.3-debug

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common

CPP_FLAGS=-I/Users/spott/Code/include/ -I/Users/spott/Code/libs/petsc-3.5.3/include/ -I./include/ -std=c++1y -Wall -Wpedantic -Werror
LDFLAGS=


CSOURCE=src/petsc_cpp/HermitianTranspose.c
CHEADER=include/petsc_cpp/HermitianTranspose.h
SOURCES=src/petsc_cpp/Matrix.cpp src/petsc_cpp/Vector.cpp src/petsc_cpp/EigenvalueSolver.cpp src/petsc_cpp/Utils.cpp test.cpp
HEADERS=include/petsc_cpp/Matrix.hpp include/petsc_cpp/Vector.hpp include/petsc_cpp/Petsc.hpp include/petsc_cpp/Utils.hpp include/petsc_cpp/EigenvalueSolver.hpp
OBJECTS=$(SOURCES:.cpp=.o)
COBJECT=$(CSOURCE:.c=.o)
EXECUTABLE=test
LIB=lib/libpetsc_cpp.a
DEFAULT=all

all: format $(EXECUTABLE) library
	echo "It compiles!  You should commit!"

${COBJECT}:
	mpicc src/petsc_cpp/HermitianTranspose.c -o src/petsc_cpp/HermitianTranspose.o -c -I/Users/spott/Code/libs/petsc-3.5.3/include/ -I${PETSC_DIR}/include/

${EXECUTABLE}: ${OBJECTS} ${COBJECT} chkopts
	echo ${OBJECTS}
	${CLINKER} -o ${EXECUTABLE} ${COBJECT} ${OBJECTS} ${LDFLAGS} ${PETSC_VEC_LIB} ${SLEPC_LIB}

library: ${OBJECTS} ${COBJECT}
	ar rus ${LIB} ${COBJECT} ${OBJECTS}

syntax_check: chkopts
	clang++ -fsyntax-only ${SOURCES} ${CPP_FLAGS} -I${SLEPC_DIR}/include/ -I${PETSC_DIR}/include/

format:
	clang-format -style=file -i ${SOURCES}
	clang-format -style=file -i ${HEADERS}
