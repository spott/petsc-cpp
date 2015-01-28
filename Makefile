#include ../makefile.include


PETSC_DIR=/usr/local/Cellar/petsc/3.5.2-debug
SLEPC_DIR=/usr/local/Cellar/slepc/3.5.1-debug

#include ${PETSC_DIR}/conf/variables
#include ${PETSC_DIR}/conf/rules
include ${SLEPC_DIR}/conf/slepc_common

CPP_FLAGS= -I/Users/spott/Code/include/ -I. -std=c++1y
LDFLAGS=

SOURCES=src/petsc_cpp/Matrix.cpp src/petsc_cpp/Vector.cpp src/petsc_cpp/EigenvalueSolver.cpp src/petsc_cpp/Utils.cpp test.cpp
HEADERS=include/petsc_cpp/Matrix.hpp include/petsc_cpp/Vector.hpp include/petsc_cpp/Petsc.hpp include/petsc_cpp/Utils.hpp include/petsc_cpp/EigenvalueSolver.hpp
OBJECTS=$(SOURCES:.cpp=.o)
EXECUTABLE=test
DEFAULT=all

all: $(SOURCES) $(EXECUTABLE)

${EXECUTABLE}: ${OBJECTS}  chkopts
	-${CLINKER} -o ${EXECUTABLE} ${OBJECTS} ${LDFLAGS} ${PETSC_VEC_LIB} ${SLEPC_LIB}

syntax_check: chkopts
	-${CLINKER} -fsyntax-only ${SOURCES} -I${SLEPC_DIR}/include/ -I./include/ -std=c++1y -I${PETSC_DIR}/include/

format:
	clang-format -style=file -i ${SOURCES}
	clang-format -style=file -i ${HEADERS}
#.PHONEY clean
#clean:
#	rm *.o;
#	rm src/petsc_cpp/*.o
