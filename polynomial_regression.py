import pandas as pd
import numpy as np

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score


# ============================================================
# CONFIGURATION
# ============================================================

ROLL_NO = "BT2024126"

RANDOM_STATE = 42

# Polynomial degrees allowed by the assignment
VAR1_DEGREES = range(1, 11)   # up to degree 10
VAR2_DEGREES = range(1, 21)   # up to degree 20

# Ridge regularization strengths to test
ALPHAS = np.logspace(-6, 3, 19)


# ============================================================
# FILE PATHS
# ============================================================

TRAIN_VAR1 = f"{ROLL_NO}_train_var1.csv"
TEST_VAR1 = f"{ROLL_NO}_test_var1.csv"

TRAIN_VAR2 = f"{ROLL_NO}_train_var2.csv"
TEST_VAR2 = f"{ROLL_NO}_test_var2.csv"


# ============================================================
# CREATE REGULARIZED POLYNOMIAL MODEL
# ============================================================

def create_model():
    return Pipeline([
        (
            "polynomial_features",
            PolynomialFeatures(
                include_bias=False
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "ridge",
            Ridge()
        )
    ])


# ============================================================
# FIND BEST DEGREE + REGULARIZATION
# ============================================================

def find_best_model(X, y, degree_range):

    model = create_model()

    parameter_grid = {
        "polynomial_features__degree": list(degree_range),
        "ridge__alpha": ALPHAS
    }

    kfold = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=kfold,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        refit=True
    )

    grid_search.fit(X, y)

    best_model = grid_search.best_estimator_

    # Cross-validated predictions for reporting
    cv_predictions = cross_val_predict(
        best_model,
        X,
        y,
        cv=kfold,
        n_jobs=-1
    )

    mse = mean_squared_error(
        y,
        cv_predictions
    )

    r2 = r2_score(
        y,
        cv_predictions
    )

    return (
        best_model,
        grid_search.best_params_,
        mse,
        r2
    )


# ============================================================
# PHASE 1 - VAR1
# ============================================================

print("=" * 65)
print("PHASE 1 - POWER PLANT STEAM TURBINE OPTIMIZATION")
print("=" * 65)

train_var1 = pd.read_csv(TRAIN_VAR1)
test_var1 = pd.read_csv(TEST_VAR1)

X_train_var1 = train_var1.drop(columns=["y"])
y_train_var1 = train_var1["y"]

X_test_var1 = test_var1.copy()

print("\nTraining data shape:", X_train_var1.shape)
print("Test data shape:", X_test_var1.shape)

print("\nSearching polynomial degrees 1-10")
print("Searching Ridge regularization strengths...")

model_var1, params_var1, mse_var1, r2_var1 = find_best_model(
    X_train_var1,
    y_train_var1,
    VAR1_DEGREES
)

print("\nBest Var1 parameters:")
print("Polynomial degree:",
      params_var1["polynomial_features__degree"])
print("Ridge alpha:",
      params_var1["ridge__alpha"])

print("Cross-validation MSE:", mse_var1)
print("Cross-validation R2:", r2_var1)

# Final model is already refitted on the complete training set
pred_var1 = model_var1.predict(X_test_var1)

output_var1 = f"{ROLL_NO}_pred_var1.csv"

pd.DataFrame({
    "y": pred_var1
}).to_csv(
    output_var1,
    index=False
)

print("\nVar1 predictions saved to:", output_var1)


# ============================================================
# PHASE 2 - VAR2
# ============================================================

print("\n")
print("=" * 65)
print("PHASE 2 - SUBTERRANEAN THERMAL RESERVOIR MAPPING")
print("=" * 65)

train_var2 = pd.read_csv(TRAIN_VAR2)
test_var2 = pd.read_csv(TEST_VAR2)

X_train_var2 = train_var2.drop(columns=["y"])
y_train_var2 = train_var2["y"]

X_test_var2 = test_var2.copy()

print("\nTraining data shape:", X_train_var2.shape)
print("Test data shape:", X_test_var2.shape)

print("\nSearching polynomial degrees 1-20")
print("Searching Ridge regularization strengths...")

model_var2, params_var2, mse_var2, r2_var2 = find_best_model(
    X_train_var2,
    y_train_var2,
    VAR2_DEGREES
)

print("\nBest Var2 parameters:")
print("Polynomial degree:",
      params_var2["polynomial_features__degree"])
print("Ridge alpha:",
      params_var2["ridge__alpha"])

print("Cross-validation MSE:", mse_var2)
print("Cross-validation R2:", r2_var2)

# Final model is already refitted on complete training set
pred_var2 = model_var2.predict(X_test_var2)

output_var2 = f"{ROLL_NO}_pred_var2.csv"

pd.DataFrame({
    "y": pred_var2
}).to_csv(
    output_var2,
    index=False
)

print("\nVar2 predictions saved to:", output_var2)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 65)
print("FINAL RESULTS")
print("=" * 65)

print("\nVAR1")
print("Polynomial degree:",
      params_var1["polynomial_features__degree"])
print("Ridge alpha:",
      params_var1["ridge__alpha"])
print("CV MSE:", mse_var1)
print("CV R2:", r2_var1)
print("Prediction file:", output_var1)

print("\nVAR2")
print("Polynomial degree:",
      params_var2["polynomial_features__degree"])
print("Ridge alpha:",
      params_var2["ridge__alpha"])
print("CV MSE:", mse_var2)
print("CV R2:", r2_var2)
print("Prediction file:", output_var2)

print("\nDone!")