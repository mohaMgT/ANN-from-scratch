#!/bin/bash

# Set variables
SOURCE_DIR="src"
BUILD_DIR="bin"
INCLUDE_DIR="include"
EXECUTABLE="main"

# Create the build directory if it doesn't exist
mkdir -p $BUILD_DIR

# Compile the program
g++ -I $INCLUDE_DIR $SOURCE_DIR/main.cpp -o $BUILD_DIR/$EXECUTABLE

# Run the program
./$BUILD_DIR/$EXECUTABLE
