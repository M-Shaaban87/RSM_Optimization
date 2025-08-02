import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize
from statsmodels.tools import add_constant
from itertools import combinations
import io

st.set_page_config(page_title="DoE - RSM Web App", layout="wide")
st.title("Design of Experiments: Response Surface Methodology (RSM)")

uploaded_file = st.file_uploader("Upload Excel file", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    input_columns = df.columns[:-1]
    output_column = df.columns[-1]

    st.subheader("Fix Input Values (Optional)")
    fixed_inputs = {}
    for col in input_columns:
        val = st.text_input(f"{col}", "")
        if val.strip():
            try:
                fixed_inputs[col] = float(val)
            except:
                st.warning(f"Invalid input for {col}, ignoring it.")

    if st.button("Run Analysis"):
        X = df[input_columns]
        y = df[output_column]

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        X_rsm = X_scaled_df.copy()
        for col in X.columns:
            X_rsm[f"{col}^2"] = X_scaled_df[col] ** 2
        interactions = list(combinations(X.columns, 2))
        for var1, var2 in interactions:
            X_rsm[f"{var1}*{var2}"] = X_scaled_df[var1] * X_scaled_df[var2]
        X_rsm_const = add_constant(X_rsm)

        rsm_model = sm.OLS(y, X_rsm_const).fit()

        defaults = X.mean().to_dict()
        full_row = {col: fixed_inputs.get(col, defaults[col]) for col in X.columns}
        scaled_row = scaler.transform(pd.DataFrame([full_row]))[0]

        x0 = X_scaled_df.mean().values
        bounds = []
        for i, col in enumerate(X.columns):
            if col in fixed_inputs:
                bounds.append((scaled_row[i], scaled_row[i]))
                x0[i] = scaled_row[i]
            else:
                bounds.append((0, 1))

        def optimize_target(mode="max"):
            sign = -1 if mode == "max" else 1
            def objective_fn(x_scaled):
                x_dict = dict(zip(X.columns, x_scaled))
                row = pd.DataFrame([x_dict])
                for col in X.columns:
                    row[f"{col}^2"] = row[col] ** 2
                for var1, var2 in interactions:
                    row[f"{var1}*{var2}"] = row[var1] * row[var2]
                row = add_constant(row, has_constant='add')
                return sign * rsm_model.predict(row).item()

            result = minimize(objective_fn, x0, bounds=bounds)
            optimal_scaled = result.x.reshape(1, -1)
            optimal_original = scaler.inverse_transform(optimal_scaled)
            optimal_df = pd.DataFrame(optimal_original, columns=X.columns)
            optimal_df[f"Predicted {output_column}"] = sign * result.fun
            return optimal_df

        optimal_max = optimize_target("max")
        optimal_min = optimize_target("min")

        sensitivity = []
        baseline = X_scaled_df.mean().to_dict()
        for var in X.columns:
            plus = baseline.copy()
            minus = baseline.copy()
            plus[var] = min(1, plus[var] + 0.1)
            minus[var] = max(0, minus[var] - 0.1)

            def prepare_row(val_dict):
                row = pd.DataFrame([val_dict])
                for col in X.columns:
                    row[f"{col}^2"] = row[col] ** 2
                for var1, var2 in interactions:
                    row[f"{var1}*{var2}"] = row[var1] * row[var2]
                return add_constant(row, has_constant='add')

            y_plus = rsm_model.predict(prepare_row(plus)).item()
            y_minus = rsm_model.predict(prepare_row(minus)).item()
            delta = (y_plus - y_minus) / 2
            sensitivity.append({"Variable": var, "Sensitivity": delta})

        sensitivity_df = pd.DataFrame(sensitivity).sort_values(by="Sensitivity", ascending=False)

        st.success("Analysis Completed!")

        st.subheader("Optimal Max Result")
        st.dataframe(optimal_max)

        st.subheader("Optimal Min Result")
        st.dataframe(optimal_min)

        st.subheader("Sensitivity Analysis")
        st.dataframe(sensitivity_df)

        st.download_button("Download Sensitivity as Excel", data=sensitivity_df.to_excel(index=False), file_name="sensitivity.xlsx")

        st.subheader("2D/3D Plot")
        var1 = st.selectbox("X Variable", input_columns)
        var2 = st.selectbox("Y Variable", input_columns)
        plot_type = st.radio("Plot Type", ["2D", "3D"])

        if st.button("Show Plot"):
            V1, V2 = np.meshgrid(np.linspace(0, 1, 30), np.linspace(0, 1, 30))
            Z = np.zeros_like(V1)

            model = sm.OLS(y, add_constant(X_rsm)).fit()
            baseline = X_scaled_df.mean()

            for i in range(V1.shape[0]):
                for j in range(V1.shape[1]):
                    row = baseline.copy()
                    row[var1] = V1[i, j]
                    row[var2] = V2[i, j]
                    row_df = pd.DataFrame([row])
                    for col in X.columns:
                        row_df[f"{col}^2"] = row_df[col] ** 2
                    for a, b in interactions:
                        row_df[f"{a}*{b}"] = row_df[a] * row_df[b]
                    row_df = add_constant(row_df, has_constant='add')
                    Z[i, j] = model.predict(row_df).item()

            fig = plt.figure(figsize=(8, 6))
            if plot_type == "3D":
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(V1, V2, Z, cmap='viridis')
                ax.set_zlabel(output_column)
            else:
                ax = fig.add_subplot(111)
                contour = ax.contourf(V1, V2, Z, levels=20, cmap='viridis')
                fig.colorbar(contour)

            ax.set_xlabel(var1)
            ax.set_ylabel(var2)
            ax.set_title(f"{plot_type} Plot: {var1} vs {var2}")

            st.pyplot(fig)

    st.markdown("""
    **Developed by**  
    Dr. Mohammed Shaaban Selim, Ph.D. (Structural Engineering)  
    mohamed.selim@deltauniv.edu.eg
    """)
