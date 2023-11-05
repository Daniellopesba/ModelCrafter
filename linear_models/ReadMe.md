
# Least Squares Residuals Analysis

## Overview

Using the least squares method to calculate the best-fit line for a given dataset.
This method works by minimizing the total of the squared differences between the actual data points
and the ones predicted by the line.

## Problem Statement

* Each data point is represented as `(x_i, y_i)`.
* The vertical distance from the point to the line `y = ax + b` is termed as the residual `R`.
* The goal is to minimize the sum of these squared residuals (SSR) across all data points.


## Calculating Residuals

* The residual `R` for a data point `(x_i, y_i)` is given by the equation: `R = y_i - (ax_i + b)`.
* Sum of Squared Residuals (SSR) is calculated as:

  ```math
  SSR = ∑(y_i - (ax_i + b))^2
  ```

## Minimization Objective

To determine the best-fitting line, we seek to minimize SSR, which involves finding the values of `a` (slope) and `b` (y-intercept) that lead to the smallest SSR.

## Analytical Solution Approach

* Expand the SSR to formulate the objective function.
* Set the partial derivatives of SSR with respect to `a` and `b` to zero to find the minimum SSR.

### Deriving the Normal Equations

Through the minimization process, we derive the normal equations which can be solved to find `a` and `b`:

```math
a = \frac{\sum{x_iy_i} - b \sum{x_i}}{\sum{x_i^2}}
  ```

* Equation for `b`:

  ```math
  a∑x_i + mb = ∑y_i
  ```

### Simplification

Solving the above equations, we get:

* Simplified form for `a`:

  ```math
  a = (mS_xy - S_xS_y) / (mS_xx - S_x^2)
  ```

* Simplified form for `b`:

  ```math
  b = (S_y - aS_x) / m
  ```
