import numpy as np
import streamlit as st

def compute_g_matrix(m, n, p, q, W):
    g_matrix = np.zeros((m*n, p*q), dtype=complex)
    
    for i in range(m):
        for j in range(n):
            for k in range(p):
                for l in range(q):
                    row = i * n + j
                    col = k * q + l
                    angle = 2 * np.pi / W * (k * (i + 1) + l * (j + 1))
                    g_matrix[row, col] = np.cos(angle) - 1j * np.sin(angle)
    
    return g_matrix

def compute_complex_conjugate(matrix):
    return np.conjugate(matrix)

def compute_transpose(matrix):
    return np.transpose(matrix)

# Streamlit app
st.title("Matrix Computation")

st.sidebar.header("Input Parameters")
m = st.sidebar.number_input("Enter m:", min_value=1, value=3)
n = st.sidebar.number_input("Enter n:", min_value=1, value=3)
p = st.sidebar.number_input("Enter p:", min_value=1, value=3)
q = st.sidebar.number_input("Enter q:", min_value=1, value=3)
W = st.sidebar.number_input("Enter W:", min_value=1, value=3)

if st.sidebar.button("Compute"):
    g_matrix = compute_g_matrix(m, n, p, q, W)
    gcc_matrix = compute_complex_conjugate(g_matrix)
    gcc_transpose = compute_transpose(gcc_matrix)

    st.write("### g matrix")
    st.write(g_matrix)

    st.write("### Complex conjugate matrix (gcc)")
    st.write(gcc_matrix)

    st.write("### Transpose of gcc matrix")
    st.write(gcc_transpose)

if __name__ == "__main__":
    st.write("Adjust the parameters in the sidebar and click 'Compute' to see the results.")
