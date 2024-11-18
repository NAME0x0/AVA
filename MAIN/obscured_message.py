import os
import random
import time
import ctypes

# Function to print the obscured message
def print_obscured_message():
    obscured_message = "HVG lv NBZF. Wzmt zqrzehs jvwu. Tdx pft zcm wqrzq smejv oaa shqz rwzhs. Ixv wftt'c fzttgsn op qx zviyvw yw uzwq mfi gtvg vfgxvwmw."
    print("\nPrinting obscured message:")
    print(obscured_message)

# Function to create a matrix-like background
def create_matrix_background(width, height):
    print("\nCreating matrix-like background:")
    for _ in range(height):
        for _ in range(width):
            # Randomly select a character from a predefined list
            print(random.choice(['0', '1', ' ']), end='')
        print()

# Function to open a window with the specified dimensions
def open_window(width, height):
    print("\nOpening window with dimensions:", width, "x", height)
    # Use ctypes to set the console window size
    os.system(f"mode con: cols={width} lines={height}")

def main():
    # Dimensions for the matrix-like background
    matrix_width = 30
    matrix_height = 20

    # Print the obscured message
    print_obscured_message()

    # Create the matrix-like background
    create_matrix_background(matrix_width, matrix_height)

    # Open a window with the specified dimensions
    open_window(matrix_width, matrix_height)

if __name__ == "__main__":
    main()
