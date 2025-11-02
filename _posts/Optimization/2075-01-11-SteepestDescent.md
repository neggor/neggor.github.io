---
layout: post
title: "Steepest descent: A geometrical take from gradient descent to Newton's method."
tag: ML
---

<link href="/css/syntax.css" rel="stylesheet">

# Motivation

# Theory

## Gradient descent with exact line search: convergence analysis
A key element in understanding gradient descent is its relationship with the condition number of the Hessian of the function ("loss") we are minimizing. This will give an interesting framework to understand where Newton's method advantages. Let’s follow §9.3.1 from [1]. 

Consider the gradient descent algorithm with exact line search, let $f(x): \mathbb{R^n} \rightarrow \mathbb{R}$:

```text
Algorithm: Gradient descent with exact line search

Given x0 in R^n, tolerance epsilon > 0

for k = 0, 1, 2, ...:
    g_k = grad f(x_k)
    if ||g_k|| <= epsilon:
        return x_k
    alpha_k = argmin_{alpha > 0} f(x_k - alpha*g_k)
    x_{k+1} = x_k - alpha_k * g_k
```

Note that this involves solving:

$$
\alpha_k = \arg\min_{\alpha > 0} f(\mathbf{x_k} - \alpha \mathbf{g_k}).
$$

If $f(x)$ is a linear function there is no finite minimum so there is no real interest here. For the quadratic case there [exist an analytical solution](#line-search), but in general there is not and line search algorithms will be used. Let us assume that the exact optimal step is available.

(SHow the proof to arrive to the condition number)

## Steepest descent
(just describe it using the idea of the norm, maybe even start by the proof that gradient descent is the steepest descent direction with the dot product formula in terms of the cosine)
(Note that this requires some lemmas that I will not prove here, essentially the upper and lower bound of the hessian eigenvalue)

## Newton’s Method
(Describe starting wiht the idea of root finding and arriving to the optimziation with the steepest descent "quadratic norm")

# Interesting Examples

## Linear models

## Re-examining OLS

## Iteratively Reweighted Least Squares

## Lagrange Multipliers

# Some Notes on Modern Algorithms

## Quasi-Newton Methods

## Adam


<details id="line-search">
    <summary>On quadratic forms and exact solution for line search</summary>
    Does $\mathbf{A}$ need to be symmetric?:  <br>
    Since $\mathbf{x^\top Ax}$ is an scalar, it must be that $\mathbf{x^\top Ax} = \big(\mathbf{x^\top Ax}\big)^\top = \mathbf{x^\top A^\top x}$. Then it follows that $\mathbf{x^\top Ax} = \mathbf{x^\top \big((A + A^\top) /2 \big) x}$. Hence, we can always substitute $\mathbf{A}$ by $\mathbf{A_s} = \mathbf{\big((A + A^\top) /2 \big)}$ and obtain the same scalar. Since $\mathbf{(A + A^\top) /2}$ is symmetric, it is safe to assume that, for any quadratic form, $\mathbf{A}$ is symmetric.
    <br>
    PD and PSD: <br>
    (Obtain the close form solution, show that if PSD we might have several minimizers, which is fine)
</details>


---

## References

These notes are based on:

1. **Boyd, S., & Vandenberghe, L.**  
   *[Convex Optimization](https://stanford.edu/~boyd/cvxbook/)*
2. **Carl D. Meyer**  
    *[Matrix Analysis and Applied Linear Algebra](https://www.stat.uchicago.edu/~lekheng/courses/309/books/Meyer.pdf)*

