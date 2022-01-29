BIN=Main
OBJ_DIR = objs
MKDIR = mkdir -p

EIGEN_DIR = /home/maeda/univ/interaction/lib/eigen
BULLET_INC_DIR = /home/maeda/univ/interaction/lib/bullet3-2.87/src

PYTHON_INC = /home/maeda/univ/interaction/dmenv/include/python3.6m
PYTHON_LIB = /home/maeda/univ/interaction/dmenv/lib/python3.6

NUMPY_INC = /home/maeda/univ/interaction/dmenv/lib/python3.6/site-packages/numpy/core/include

INC = -I./ \
	-I$(EIGEN_DIR) \
	-I$(BULLET_INC_DIR) \
	-I$(PYTHON_INC) \
	-I$(NUMPY_INC) 

LIBS = -lGLEW -lGL -lGLU -lglut -lBulletDynamics -lBulletCollision -lLinearMath -lm -lstdc++ -lpython3.6m -ljvm

CC = clang++
#CC = g++
CFLAGS = -std=c++11 -O3 -fPIC #-Wall

SRC_DIRS = util/ \
	util/json/ \
	anim/ \
	sim/ \
	render/ \
	render/lodepng/ \
	scenes/ \
	scenes/arm/ \
	scenes/particle/ \
	scenes/pendulum/

SRC_FILES = $(BIN).cpp DeepMimicCore.cpp $(foreach dir, $(SRC_DIRS), $(wildcard $(dir)*.cpp))
OUT_DIRS = $(addprefix $(OBJ_DIR)/, $(SRC_DIRS))
OBJ = $(addprefix $(OBJ_DIR)/, $(patsubst %.cpp, %.o, $(SRC_FILES)))


default: build

build: $(BIN)

print:
	$(OUT_DIRS)

python: $(OUT_DIRS) $(OBJ)
	swig -c++ -python DeepMimicCore.i
	$(CC) -c -g $(CFLAGS) $(INC) $(LIBS) DeepMimicCore.cpp DeepMimicCore_wrap.cxx -I$(PYTHON_INC)
	$(CC) -shared $(CFLAGS) $(OBJ) $(LIBS) DeepMimicCore_wrap.o -o _DeepMimicCore.so -L$(PYTHON_LIB) -L$(JAVA_HOME)/lib/server

$(OBJ_DIR)/%.o: %.cpp
	$(CC) -c -g $(CFLAGS) $(INC) -o $@ $<

$(BIN): $(OUT_DIRS) $(OBJ)
	$(CC) -g $(CFLAGS) $(OBJ) $(LIBS) -o $@

$(OUT_DIRS):
	$(MKDIR) $(OUT_DIRS)
	
clean:
	- rm -f -r $(OUT_DIRS) $(BIN) $(OBJ) $(OBJ_DIR) DeepMimicCore_wrap.cxx _DeepMimicCore.so
