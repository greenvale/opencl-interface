PROGRAM_NAME := a
BIN_DIR := .
INCLUDE_DIR := .
LIB_DIR := .
SOURCE_DIR := .
BUILD_DIR := .

LIBS := -L. -lOpenCL
INCLUDES := -I.

$(BIN_DIR)/$(PROGRAM_NAME): main.o OpenCL_Interface.o OpenCL_KernelInterface.o
	g++ -o $(BIN_DIR)/$(PROGRAM_NAME) $^ $(LIBS)

main.o: main.cpp
	g++ $(INCLUDES) -c -g main.cpp

OpenCL_Interface.o: OpenCL_Interface.cpp
	g++ $(INCLUDES) -c -g OpenCL_Interface.cpp

OpenCL_KernelInterface.o: OpenCL_KernelInterface.cpp
	g++ $(INCLUDES) -c -g OpenCL_KernelInterface.cpp

clean:
	rm $(BIN_DIR)/$(PROGRAM_NAME) $(BUILD_DIR)/*.o