PROGRAM_NAME := app
BIN_DIR := .
INCLUDE_DIR := .
LIB_DIR := .
SOURCE_DIR := .
BUILD_DIR := .

LIBS := -L. -lOpenCL
INCLUDES := -I.

$(BIN_DIR)/$(PROGRAM_NAME): main.o OpenCLInterface.o
	g++ -o $(BIN_DIR)/$(PROGRAM_NAME) $^ $(LIBS)

main.o: main.cpp
	g++ $(INCLUDES) -c -g main.cpp

OpenCLInterface.o: OpenCLInterface.cpp
	g++ $(INCLUDES) -c -g OpenCLInterface.cpp

clean:
	rm $(BIN_DIR)/$(PROGRAM_NAME) $(BUILD_DIR)/*.o