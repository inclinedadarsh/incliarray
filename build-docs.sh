#!/bin/bash

# Clone the Doxygen Awesome CSS repository
echo "Cloning doxygen-awesome-css..."
git clone https://github.com/jothepro/doxygen-awesome-css.git

# Run Doxygen to generate documentation
echo "Running Doxygen..."
doxygen Doxyfile

# Clean up the cloned repository
echo "Cleaning up..."
rm -rf doxygen-awesome-css

echo "Doxygen documentation built successfully in the 'html' directory."
