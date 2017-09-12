CPP_FILES = $(notdir $(wildcard src/*/*.cpp))
OBJ_FILES = $(addprefix obj/, $(CPP_FILES:.cpp=.o))
INC=-Iinclude
TEST_FILES = $(wildcard test/cpp/*.cpp)

VPATH = src:src/base:src/dataset:src/smooth_loss:src/sparse_linear

FLAGS = -O3 -Wall

LIBNAME=blitzml
SLIB = lib$(LIBNAME).so

all: $(SLIB) 

$(SLIB): $(OBJ_FILES) | lib
	$(CXX) -shared -o lib/$@ $^ 

obj/%.o: %.cpp
	$(CXX) $(FLAGS) -fPIC $(INC) -c -o $@ $<

$(OBJ_FILES): | obj

obj:
	mkdir -p obj

lib:
	mkdir -p lib

runtest: $(TEST_FILES) | $(SLIB)
	g++ -o test/cpp/run -g $^ $(INC) -Llib -l$(LIBNAME)
	LD_LIBRARY_PATH=$(LD_LIBRARY_PATH):./lib ./test/cpp/run

clean:
	rm -rf obj
	rm -rf lib
	rm -rf test/cpp/run*

