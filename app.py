import numpy as np
import streamlit as st

# Function to compute a single element of the g matrix
def compute_g_element(m, n, p, q, W):
    angle = 2 * np.pi / W * (p * m + q * n)  # Calculate the angle for the trigonometric functions
    g_element = np.cos(angle) - 1j * np.sin(angle)  # Compute the complex value for the g matrix element
    return g_element

# Function to compute the g matrix (as a single-element matrix)
def compute_g_matrix(m, n, p, q, W):
    g_matrix = np.zeros((1, 1), dtype=complex)  # Initialize the g matrix with a single element
    g_matrix[0, 0] = compute_g_element(m, n, p, q, W)  # Compute and assign the single element
    return g_matrix

# Function to compute the complex conjugate of a matrix
def compute_complex_conjugate(matrix):
    return np.conjugate(matrix)

# Function to compute the transpose of a matrix
def compute_transpose(matrix):
    return np.transpose(matrix)

# Streamlit app setup
st.title("Single Element Matrix Computation")  # Title of the Streamlit app

# Sidebar for input parameters
st.sidebar.header("Input Parameters")  # Header for the sidebar
m = st.sidebar.number_input("Enter m:", min_value=1, value=3)  # Input for m
n = st.sidebar.number_input("Enter n:", min_value=1, value=3)  # Input for n
p = st.sidebar.number_input("Enter p:", min_value=1, value=3)  # Input for p
q = st.sidebar.number_input("Enter q:", min_value=1, value=3)  # Input for q
W = st.sidebar.number_input("Enter W:", min_value=1, value=3)  # Input for W

# Button to trigger the computation
if st.sidebar.button("Compute"):
    # Compute the g matrix using the provided input parameters
    g_matrix = compute_g_matrix(m, n, p, q, W)
    # Compute the complex conjugate of the g matrix
    gcc_matrix = compute_complex_conjugate(g_matrix)
    # Compute the transpose of the complex conjugate matrix
    gcc_transpose = compute_transpose(gcc_matrix)

    # Display the computed matrices in the main app area
    st.write("### g matrix")
    st.write(g_matrix)

    st.write("### Complex conjugate matrix (gcc)")
    st.write(gcc_matrix)

    st.write("### Transpose of gcc matrix")
    st.write(gcc_transpose)

# Main function to run the app (not strictly necessary for Streamlit, but good practice)
if __name__ == "__main__":
    st.write("Adjust the parameters in the sidebar and click 'Compute' to see the results.")
