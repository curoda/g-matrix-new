import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Function to compute a single element of the g matrix
def compute_g_element(m, n, p, q, W):
    angle = 2 * np.pi / W * (p * m + q * n)  # Calculate the angle for the trigonometric functions
    g_element = np.cos(angle) - 1j * np.sin(angle)  # Compute the complex value for the g matrix element
    return g_element

# Function to compute the g matrix for multiple sets of m, n, p, q values
def compute_g_matrix(mn_params, pq_params, W):
    num_rows = len(mn_params)
    num_cols = len(pq_params)
    
    g_matrix = np.zeros((num_rows, num_cols), dtype=complex)
    
    for col, (p, q) in enumerate(pq_params):
        for row, (m, n) in enumerate(mn_params):
            g_matrix[row, col] = compute_g_element(m, n, p, q, W)
    
    return g_matrix

# Function to compute the complex conjugate of a matrix
def compute_complex_conjugate(matrix):
    return np.conjugate(matrix)

# Function to compute the transpose of a matrix
def compute_transpose(matrix):
    return np.transpose(matrix)

# Streamlit app setup
st.title("Matrix Computation for Multiple Values")  # Title of the Streamlit app

# Sidebar for input parameters
st.sidebar.header("Input Parameters")  # Header for the sidebar

# Text area to input multiple values of m, n, p, q
param_input = st.sidebar.text_area(
    "Enter m, n, p, q values (one set per line, separated by spaces):",
    "20 30 98 17\n24 29 96 28\n27 27 94 34\n29 24 91 42\n30 20 87 50\n29 16 83 56\n27 13 79 62\n24 11 74 67\n20 10 67 74\n16 11 62 79\n13 13 56 83\n11 16 50 87\n10 20 42 91\n11 24 34 94\n13 27 28 96\n16 29 17 98"
)
W = st.sidebar.number_input("Enter W:", min_value=1, value=10000)  # Default value of W set to 10000

# File uploader for the V matrix
v_file = st.sidebar.file_uploader("Upload the V matrix (Excel file)", type=["xlsx"])

# File uploader for the Velocity matrix
velocity_file = st.sidebar.file_uploader("Upload the Velocity matrix (Excel file)", type=["xlsx"])

# Button to trigger the computation
if st.sidebar.button("Compute"):
    # Parse the input values
    param_lines = param_input.strip().split('\n')
    all_params = [tuple(map(int, line.split())) for line in param_lines]

    mn_params = [(m, n) for m, n, _, _ in all_params]
    pq_params = [(p, q) for _, _, p, q in all_params]

    # Compute the g matrix for all sets of parameters
    g_matrix = compute_g_matrix(mn_params, pq_params, W)

    # Compute the complex conjugate and transpose for the g matrix
    gcc_matrix = compute_complex_conjugate(g_matrix)
    gcc_transpose = compute_transpose(gcc_matrix)

    if v_file and velocity_file:
        # Read the V matrix from the uploaded Excel file
        v_matrix = pd.read_excel(v_file, header=None).to_numpy()
        
        # Read the Velocity matrix from the uploaded Excel file
        velocity_matrix = pd.read_excel(velocity_file, header=None).to_numpy()

        # Check if dimensions match for matrix multiplication
        if v_matrix.shape[0] != g_matrix.shape[1] or v_matrix.shape[1] != g_matrix.shape[1]:
            st.error("The dimensions of the V matrix do not match the required dimensions for multiplication.")
        elif velocity_matrix.shape[0] != g_matrix.shape[0] or velocity_matrix.shape[1] != 1:
            st.error("The dimensions of the Velocity matrix do not match the required dimensions for multiplication.")
        else:
            # Compute Matrix Z
            z_matrix = np.dot(np.dot(g_matrix, v_matrix), gcc_transpose)
            
            # Compute the Pressure Column Vector
            pressure_column_vector = np.dot(z_matrix, velocity_matrix)

            # Separate the real and imaginary components
            real_pressure = np.real(pressure_column_vector)
            imag_pressure = np.imag(pressure_column_vector)

            # Compute the arctan of the imaginary component over the real component
            atan_matrix = np.arctan2(imag_pressure, real_pressure)

            # Combine into a DataFrame
            pressure_df = pd.DataFrame({
                'Real': real_pressure.flatten(),
                'Imaginary': imag_pressure.flatten()
            })

            atan_df = pd.DataFrame(atan_matrix, columns=['ATAN'])

            # Plot the ATAN values
            fig, ax = plt.subplots()
            ax.plot(atan_df.index, atan_df['ATAN'], marker='o')
            ax.set_title('ATAN (arctan(Imaginary / Real))')
            ax.set_xlabel('Index')
            ax.set_ylabel('ATAN Value')
            st.pyplot(fig)

            # Display the computed matrices in the main app area
            st.write("### g matrix")
            st.write(pd.DataFrame(g_matrix))

            st.write("### Complex conjugate matrix (gcc)")
            st.write(pd.DataFrame(gcc_matrix))

            st.write("### Transpose of gcc matrix")
            st.write(pd.DataFrame(gcc_transpose))

            st.write("### V matrix")
            st.write(pd.DataFrame(v_matrix))

            st.write("### Z matrix (G * V * Transpose(gcc))")
            st.write(pd.DataFrame(z_matrix))
            
            st.write("### Velocity Matrix")
            st.write(pd.DataFrame(velocity_matrix))

            st.write("### Pressure Column Vector (Z * Velocity)")
            st.write(pressure_df)

            st.write("### ATAN (arctan(Imaginary / Real))")
            st.write(atan_df)
    else:
        st.error("Please upload both the V matrix and the Velocity matrix to compute Matrix Z and the Pressure Column Vector.")

# Main function to run the app (not strictly necessary for Streamlit, but good practice)
if __name__ == "__main__":
    st.write("Enter the parameters in the sidebar and click 'Compute' to see the results.")
