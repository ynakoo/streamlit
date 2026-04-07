import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("Interactive Linear Regression")

# Sidebar controls
m = st.sidebar.slider("Slope (m)", -5.0, 5.0, 1.0)
b = st.sidebar.slider("Intercept (b)", -10.0, 10.0, 0.0)
noise = st.sidebar.slider("Noise", 0.0, 5.0, 1.0)
lr = st.sidebar.slider("Learning Rate", 0.001, 1.0, 0.01)
iterations = st.sidebar.slider("Iterations", 1, 100, 10)
add_outliers = st.sidebar.checkbox("Add Outliers")

# Generate data
np.random.seed(42)
X = np.linspace(0, 10, 50)
y = 2 * X + 3 + np.random.randn(50) * noise

# ✅ Apply outliers BEFORE training
if add_outliers:
    y[::5] += np.random.randn(len(y[::5])) * 10

# Prediction
y_pred = m * X + b

# Gradient Descent
m_gd, b_gd = 0, 0
for _ in range(iterations):
    y_pred_gd = m_gd * X + b_gd
    dm = (-2/len(X)) * np.sum(X * (y - y_pred_gd))
    db = (-2/len(X)) * np.sum(y - y_pred_gd)
    m_gd -= lr * dm
    b_gd -= lr * db

# Plot (ALL things before showing)
fig, ax = plt.subplots()
ax.scatter(X, y, label="Data")
ax.plot(X, y_pred, color='red', label="Manual Line")
ax.plot(X, m_gd * X + b_gd, color='green', label="Gradient Descent")

# Residuals
for i in range(len(X)):
    ax.plot([X[i], X[i]], [y[i], y_pred[i]], color='gray', alpha=0.5)

ax.legend()
st.pyplot(fig)

# MSE
mse = np.mean((y - y_pred) ** 2)
st.write(f"### MSE: {mse:.2f}")

# Loss vs Iterations
losses = []
m_temp, b_temp = 0, 0

for _ in range(iterations):
    y_pred_temp = m_temp * X + b_temp
    loss = np.mean((y - y_pred_temp) ** 2)
    losses.append(loss)

    dm = (-2/len(X)) * np.sum(X * (y - y_pred_temp))
    db = (-2/len(X)) * np.sum(y - y_pred_temp)

    m_temp -= lr * dm
    b_temp -= lr * db

st.line_chart(losses)

# Loss Surface
m_vals = np.linspace(-3, 5, 50)
b_vals = np.linspace(-5, 10, 50)

loss_grid = np.zeros((len(m_vals), len(b_vals)))

for i, m_ in enumerate(m_vals):
    for j, b_ in enumerate(b_vals):
        y_pred_ = m_ * X + b_
        loss_grid[i, j] = np.mean((y - y_pred_)**2)

st.write("Loss Surface Heatmap")

# ✅ FIXED (matplotlib instead of st.imshow)
fig2, ax2 = plt.subplots()
c = ax2.imshow(loss_grid, cmap='viridis')
fig2.colorbar(c)

st.pyplot(fig2)

# Info
st.info("Linear regression tries to find the best line minimizing squared error.")