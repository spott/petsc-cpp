include ${SLEPC_DIR}/lib/slepc-conf/slepc_variables


release_cpp_flags = -Wall -Wpedantic -Wextra -fdiagnostics-color=auto
clang_cpp_flags   = -fdiagnostics-show-template-tree -Wbind-to-temporary-copy -Weverything -D_DEBUG
debug_cpp_flags   = ${release_cpp_flags} -Werror -Wno-c++98-compat-pedantic \
					          -Wno-old-style-cast -Wno-padded -Wno-deprecated-declarations
CPP_FLAGS         = -I./include/ -std=c++1y ${SLEPC_CC_INCLUDES} ${PETSC_CC_INCLUDES}
LD_FLAGS          = ${SLEPC_LIB} ${PETSC_LIB}

#Directories:
source   = ./src/petsc_cpp
includes = ./include/petsc_cpp
build    = ./build

modules      = Matrix.cpp Vector.cpp EigenvalueSolver.cpp Utils.cpp Draw.cpp
objects      = ${patsubst %.cpp, ${build}/%.o, ${modules}}
library_file = lib/libpetsc_cpp.a



clang: CXX=clang++
clang: CPP_FLAGS += ${clang_cpp_flags} ${debug_cpp_flags}
clang: library test

gpp: CXX=g++-4.9
gpp: CPP_FLAGS += ${debug_cpp_flags}
gpp: library test

release: CPP_FLAGS +=${RELEASE_FLAGS}
release: library

test: ${objects} test.cpp
	mpicxx -o $@ $^ ${LD_FLAGS} ${CPP_FLAGS}

library: ${objects} 
	@mkdir -p ${dir ${library_file}}
	ar rus ${library_file} $^

${build}/%.o: ${source}/%.cpp
	@mkdir -p ${dir $@}
	mpicxx -o $@ -c $< ${CPP_FLAGS}

${source}/%.cpp: ${includes}/%.hpp ${includes}/Petsc.hpp
	-clang-format -style=file -i $@

${includes}/%.hpp:
	-clang-format -style=file -i $@ $<

syntax_check: chkopts
	clang++ -fsyntax-only ${SOURCES} ${CPP_FLAGS_} ${CLANG_ONLY_FLAGS} ${DEBUG_FLAGS} ${SLEPC_CC_INCLUDES}

variables:
	@echo ${objects}

clean:
	rm -rf ${build}
	rm -f ${library_file}

.PHONEY: clean library
