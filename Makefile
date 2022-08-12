PROGRAM_NAME := app
BIN_DIR := .
INCLUDE_DIR := .
LIB_DIR := .
SOURCE_DIR := .
BUILD_DIR := .

LIBS := -L. -lOpenCL
INCLUDES := -I.

$(BIN_DIR)/$(PROGRAM_NAME): main.o OpenCLApp.o
	g++ -o $(BIN_DIR)/$(PROGRAM_NAME) $^ $(LIBS)

main.o: main.cpp
	g++ $(INCLUDES) -c -g main.cpp

OpenCLApp.o: OpenCLApp.cpp
	g++ $(INCLUDES) -c -g OpenCLApp.cpp

clean:
	rm $(BIN_DIR)/$(PROGRAM_NAME) $(BUILD_DIR)/*.o