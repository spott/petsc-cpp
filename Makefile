#include ../makefile.include

PETSC_DIR=/usr/local/Cellar/petsc/3.4.2-debug
SLEPC_DIR=/usr/local/Cellar/slepc/3.4.2-debug

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common

#OMPI_CXX=g++-4.9

CPP_FLAGS=-I./include/
LDFLAGS=

SOURCES=test.cpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=test
DEFAULT=all

all: $(SOURCES) $(EXECUTABLE)

${EXECUTABLE}: ${OBJECTS}  chkopts
	-${CLINKER} -o ${EXECUTABLE} ${OBJECTS} ${LDFLAGS} ${PETSC_VEC_LIB} ${SLEPC_LIB}

