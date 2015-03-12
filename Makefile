include ${SLEPC_DIR}/conf/slepc_common


RELEASE_FLAGS=-Wall -Wpedantic -Wextra
DEBUG_FLAGS=-fdiagnostics-show-template-tree -Wall -Wpedantic -Wextra -Werror -Wbind-to-temporary-copy -Weverything -Wno-c++98-compat-pedantic -Wno-old-style-cast -Wno-error=padded
CPP_FLAGS= -I${PRIVATE_PETSC_DIR}/include/ -I./include/ -std=c++1y
LDFLAGS=


CSOURCE=src/petsc_cpp/HermitianTranspose.c
CHEADER=include/petsc_cpp/HermitianTranspose.h
SOURCES=src/petsc_cpp/Matrix.cpp src/petsc_cpp/Vector.cpp src/petsc_cpp/EigenvalueSolver.cpp src/petsc_cpp/Utils.cpp test.cpp
HEADERS=include/petsc_cpp/Matrix.hpp include/petsc_cpp/Vector.hpp include/petsc_cpp/Petsc.hpp include/petsc_cpp/Utils.hpp include/petsc_cpp/EigenvalueSolver.hpp include/petsc_cpp/TimeStepper.hpp
OBJECTS=$(SOURCES:.cpp=.o)
COBJECT=$(CSOURCE:.c=.o)
EXECUTABLE=test
LIB=lib/libpetsc_cpp.a
UNAME := $(shell uname)
ifeq ($(UNAME), Linux)
	DEFAULT=release
endif
ifeq ($(UNAME), Darwin)
	DEFAULT=debug
endif

debug: CPP_FLAGS += ${DEBUG_FLAGS}
debug: format $(EXECUTABLE) library
	echo "It compiles!  You should commit!"

release: CPP_FLAGS += ${RELEASE_FLAGS}
release: $(EXECUTABLE) library

${COBJECT}:
	mpicc src/petsc_cpp/HermitianTranspose.c -o src/petsc_cpp/HermitianTranspose.o -c -I/Users/spott/Code/libs/petsc-3.5.3/include/ -I${PETSC_DIR}/include/

${EXECUTABLE}: ${OBJECTS} ${COBJECT} chkopts
	${CLINKER} -o ${EXECUTABLE} ${COBJECT} ${OBJECTS} ${LDFLAGS} ${PETSC_VEC_LIB} ${SLEPC_LIB}

library: ${OBJECTS} ${COBJECT}
	-mkdir lib/
	ar rus ${LIB} ${COBJECT} ${OBJECTS}

syntax_check: chkopts
	clang++ -fsyntax-only ${SOURCES} ${CPP_FLAGS} -I${SLEPC_DIR}/include/ -I${PETSC_DIR}/include/

format:
	clang-format -style=file -i ${SOURCES}
	clang-format -style=file -i ${HEADERS}

cleanup:
	-rm ${OBJECTS}
	-rm ${COBJECT}
	-rm ${LIB}
