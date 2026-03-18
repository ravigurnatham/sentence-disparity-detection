# Mathematical Foundation: Predictive Justice System

This document outlines the core algorithms and statistical formulations serving as the foundation for the EB-2 NIW sentencing disparity prototype. The goal is to mathematically define fairness and isolate arbitrary judicial bias.

## 1. The Core Axiom: Separating Law from Discretion
A federal sentence ($S_{actual}$) is a combination of objective facts defined by the U.S. Sentencing Guidelines ($S_{guidelines}$) and a subjective adjustment applied by the judge based on jurisdiction or individual philosophy ($\epsilon_{bias}$).

```math
S_{actual} = S_{guidelines}(X_{legal}) + \epsilon_{bias}
```

Where:
*   $X_{legal}$ is the set of legally mandated variables (e.g., Primary Offense Type, Total Criminal History Points).
*   $\epsilon_{bias}$ represents the unexplainable variance (Disparity), which may be correlated with protected attributes (Race, Sex) or geographic location (Federal District).

Our prototype's goal is to learn the function $f(X_{legal}) \approx S_{guidelines}$ securely and empirically, and then isolate $\epsilon_{bias}$.

## 2. Phase 2 Baseline: Gradient Boosted Trees (XGBoost)
To model the highly non-linear nature of federal sentencing matrices, we utilized XGBoost (eXtreme Gradient Boosting).

The model learns an ensemble of $K$ regression trees to minimize the Mean Squared Error (MSE) objective:

```math
\text{Obj}(\theta) = \sum_{i=1}^{n} L(y_i, \hat{y}_i) + \sum_{k=1}^{K} \Omega(f_k)
```

Where:
*   **Loss Function ($L$)**: The squared residual $(y_i - \hat{y}_i)^2$, where $y_i$ is the actual sentence in months (`SENTTOT`) and $\hat{y}_i$ is the predicted baseline.
*   **Regularization ($\Omega$)**: Penalizes model complexity to prevent overfitting to the training set's specific historical quirks: 

```math
\Omega(f) = \gamma T + \frac{1}{2}\lambda||w||^2
```

### The "Legally Blind" Constraint
Crucially, our feature set $X_{train}$ **excludes** variables like Race (`NEWRACE`) and Sex (`MONSEX`). The function $f_k(x)$ is forced to minimize loss using *only* $X_{legal}$, establishing an unbiased empirical benchmark $\hat{y}$.

## 3. Quantifying Disparity (Residual Analysis)
Once the baseline model is trained, we evaluate sentences on a holdout test set to quantify the bias term ($\epsilon_{bias}$), which we define as the **Residual ($r_i$)**.

```math
r_i = y_{i(actual)} - \hat{y}_{i(predicted)}
```

*   If $r_i > 0$: The judge was significantly harsher than the objective facts dictate.
*   If $r_i \approx 0$: The judge followed the empirical norm perfectly.
*   If $r_i < 0$: The judge was substantially more lenient.

### Group-Level Disparate Impact
To prove systemic disparities for the NIW endeavor, we aggregate these residuals over specific protected groups ($G$):

```math
\text{Mean Disparity}(G) = \frac{1}{|G|} \sum_{i \in G} (y_i - \hat{y}_i)
```

For example, our EDA proved that the Mean Disparity for $G = \text{District 11}$ is **+35.3 months**, while for $G = \text{District 94}$ it is **-31.4 months**. 

## 4. Phase 3 Proposal: Hierarchical GLMMs
While XGBoost provides excellent non-linear point predictions, a true statistical breakdown of variance requires a **Generalized Linear Mixed Model (GLMM)** to account for the hierarchical structure of the judicial system (Defendants $\rightarrow$ Judges $\rightarrow$ Districts).

In future iterations, we will formulate the sentence length $Y_{ij}$ for defendant $i$ in district $j$ as:

```math
\log(Y_{ij}) = \beta_0 + \beta_{1}X_{ij} + u_{0j} + \epsilon_{ij}
```

Where:
*   $X_{ij}$ are the fixed effects (Offense severity, Criminal history).
*   $u_{0j} \sim \mathcal{N}(0, \sigma^2_u)$ represents the **random intercept for District $j$**.
*   $\epsilon_{ij} \sim \mathcal{N}(0, \sigma^2_e)$ is the individual-level unexplained error.

By interpreting the variance component $\sigma^2_u$, we can definitively isolate how much of the national sentencing disparity is caused entirely by geographic jurisdiction rather than the facts of the crime.
