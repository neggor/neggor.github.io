---
layout: post
title: SVM from scratch.
tag: ML
---
<link href="/css/syntax.css" rel="stylesheet" >
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


### References:
{% bibliography %}


In a previous note [note](/2024/01/16/Understanding-SVMs_1.html), I derived the details of the optimal separating hyperplance. However, I stopped at the definition of the optimization problem and used a quadratic solver. Here, first, I will go over this optimization process in detail. With that, I will then follow with the code for the SVM and check it out in the MNIST dataset.

### Understanding the problem, optimal hyperplanes from scratch:

Let's re-state the optimization problem again here. After constructing the Lagrangian and leveraging the strong duality propertiy we get a "Dual ascent" kind of problem:

$$ \text{maximize}_{\boldsymbol{\alpha}} \ \text{minimize}_{\boldsymbol{\beta}, \beta_0} \left[ \frac{1}{2} \lVert \boldsymbol{\beta} \rVert^2 - \sum_{i=1}^n \alpha_i \left( y_i (\boldsymbol{\beta}^\top  \mathbf{x}_i + \beta_0) - 1 \right) \right]$$

Where the first minimization step can be computed analytically (meaning; given $$\alpha$$, I give you the optimal $$\beta$$ and $$\beta_0$$), yielding:

$$ \text{maximize}_{\boldsymbol{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{k=1}^n \alpha_i \alpha_j y_i y_k \mathbf{x}_i^\top  \mathbf{x}_k$$

$$\text{subject to} \quad \sum_{i=1}^n \alpha_i y_i = 0 \text{ and } \alpha_i \geq 0 \quad \forall \quad i = 1, \dots, n$$

Where the first constraint comes for the requirement of optimality for $$\beta_0$$ and the second from the dual feasibility.



Now, how do we optimize this? The idea is to reframe this as a quadratic program. This is actually not too difficult. Recall the most generic quadratic programming structure:

$$
\text{minimize}_{x\in\mathbb{R^n}} \frac{1}{2}x^\top Qx + c^\top x
$$

$$\text{subject to} \quad \mathbf{A}x = b$$

We can get quite close by defining 

$$Q_{ij} = y_i y_j \, \mathbf{x}_i^\top  \mathbf{x}_j \quad \text{for } i, j = 1, \dots, n$$

Changing the sign and re-aranging:
$$
\text{minimize}_{\boldsymbol{\alpha}} \quad \frac{1}{2}\boldsymbol{\alpha}^\top Q\boldsymbol{\alpha} - \boldsymbol{1}^\top \boldsymbol{\alpha}
$$

$$\text{subject to} \quad \mathbf{y}^\top  \boldsymbol{\alpha} = 0 \text{ and } \boldsymbol{\alpha} \geq 0$$

The problem is that we still have an inequality constraint here, this makes the whole issue a bit complex still. We are going to need some more bits of theory. A very nice quote from {% cite boyd2004convex %} that helps to see what is going on:


>"We can view interior-point methods as another level in the hierarchy of convex
optimization algorithms. Linear equality constrained quadratic problems are the
simplest. For these problems the KKT conditions are a set of linear equations,
which can be solved analytically. Newton’s method is the next level in the hierarchy.
We can think of Newton’s method as a technique for solving a linear equality constrained optimization problem, with twice differentiable objective, by reducing
it to a sequence of linear equality constrained quadratic problems. Interior-point
methods form the next level in the hierarchy: They solve an optimization problem
with linear equality and inequality constraints by reducing it to a sequence of linear
equality constrained problems."



#### Logarithmic barrier 
(TODO add here a picture of the barrier, add that active point methods also exists)
The idea is to add a term to the function that we are minimizing accounting for the infeasibility of violating some inequality constraint (see section 11.2 of {% cite boyd2004convex %}). In our case, we can define 

$$
I(\boldsymbol{\alpha}) = \begin{cases} 0 & \text{if } \quad  \alpha_i  \geq 0 \quad  \forall \quad  i = 1, ..., n \\ \infty & \text{otherwise} \end{cases}
$$

and just plug this in:

$$
\text{minimize}_{\boldsymbol{\alpha}} \quad \frac{1}{2} \boldsymbol{\alpha}^\top Q\boldsymbol{\alpha} - \boldsymbol{1}^\top \boldsymbol{\alpha} + I(\boldsymbol{\alpha})
$$

$$\text{subject to} \quad \mathbf{y}^\top  \boldsymbol{\alpha} = 0 $$

However, this seems quite difficult to deal with, right? The logarithmic barrier aims to approximate $$I$$ by $$\hat{I}(\boldsymbol{\alpha}) = \sum_{i=1}^n -\frac{1}{t}\log(\alpha_i)$$. For a big $$t$$, this function greatly resembles $$I$$.

Now, optimizing direclty with a very big $$t$$ might be problematic, hence we get a sequence of solutions while increasing $$t$$. The previous obtained solution is used as the starting point for the next iteration. The idea is that as $$t$$ increases, the solution converges to the optimal solution of the original problem. The sequence of optimal values derived from different values of $$t$$ is the central path. Nevertheless, we still need to know how to solve the problem with the logarithmic barrier. This is where the Newton's method comes in.

#### Newton's method with equality constraints

At each iteration we will have to solve a problem of the form:

$$
\text{minimize}_{\boldsymbol{\alpha}} \quad \frac{1}{2} \boldsymbol{\alpha}^\top Q\boldsymbol{\alpha} - \boldsymbol{1}^\top \boldsymbol
{\alpha} - \frac{1}{t}\sum_{i=1}^n \log(\alpha_i)
$$

$$
\text{subject to} \quad \mathbf{y}^\top  \boldsymbol{\alpha} = 0
$$

Note that this is not a quadratic problem anymore, however, we can iteratively solve a quadratic approximation. Note that this is still convex since the sum of convex functions is convex. Now, in each step of the Newton's method we will have to solve a quadratic problem of the form (see section 10.2 of {% cite boyd2004convex %}):

$$
\min_{\mathbf{v}} \hat{f}\mathbf{(x + v)} = f(\boldsymbol{\alpha}) + \nabla f(\boldsymbol{\alpha})^\top  \mathbf{v} + \frac{1}{2} \mathbf{v}^\top  \nabla^2 f(\boldsymbol{\alpha}) \mathbf{v}
$$

$$
\text{subject to} \quad \mathbf{y}^\top (\boldsymbol{\alpha} + \mathbf{v}) = 0
$$

This is, we are looking for the step $$\mathbf{v}$$ that minimizes the second-order Taylor approximation near starting point $$\boldsymbol{\alpha}$$ of the function $$f$$, while satisfying the equality constraint. The Lagrangian is:

$$
\mathcal{L}(\mathbf{v}, \boldsymbol{\alpha}, \boldsymbol{\lambda}) =
\frac{1}{2} \mathbf{v}^\top  \nabla^2 f(\boldsymbol{\alpha}) \mathbf{v} + \nabla f(\boldsymbol{\alpha})^\top  \mathbf{v} + \boldsymbol{\lambda}^\top  (\mathbf{y}^\top (\boldsymbol{\alpha} + \mathbf{v}))
$$


 The optimality conditions here are:

$$
\mathbf{y}^\top \mathbf{v} = - \mathbf{y}^\top \boldsymbol{\alpha}
$$

And, since we need to fulfill the equality constraint, we can re-arrange this to:
$$
\mathbf{y}^\top \mathbf{v} = 0
$$


$$
\frac{\partial \hat{f}}{\partial \mathbf{v}} = \nabla f(\boldsymbol{\alpha}) + \nabla^2 f(\boldsymbol{\alpha}) \mathbf{v} + \mathbf{y} \boldsymbol{\lambda} = 0
$$

Which can be put in matrix form as:

$$
\begin{bmatrix}
\nabla^2 f(\boldsymbol{\alpha}) & \mathbf{y} \\ \mathbf{y}^\top  & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{v}^\star \\ \boldsymbol{\lambda}^\star
\end{bmatrix}
=
\begin{bmatrix}
-\nabla f(\boldsymbol{\alpha}) \\ 0
\end{bmatrix}
$$

Where $$\mathbf{y}^\top$$ is a row vector. This is known as the KKT system. Since it is symmetric, we can use the Cholesky decomposition to solve it. This yields the optimal step $$\mathbf{v}$$. We add this step to the current point $$\boldsymbol{\alpha}$$ and repeat the process until convergence. This is:

$$
\boldsymbol{\alpha}_{k+1} = \boldsymbol{\alpha}_k + \mathbf{v}^\star
$$

Where $$\boldsymbol{\alpha}_k$$ is the current point and $$\mathbf{v}$$ is the step we just computed. The process stops when the norm of the gradient is small enough.

##### Calculating derivatives

We need to get the gradient and hessian of the function, involving the logarithmic barrier. The function we are **minimizing** is:

$$
f(\boldsymbol{\alpha}) = \frac{1}{2} \boldsymbol{\alpha}^\top Q\boldsymbol{\alpha} - \boldsymbol{1}^\top \boldsymbol{\alpha} - \frac{1}{t}\sum_{i=1}^n \log(\alpha_i)
$$

The gradient is:

$$
\nabla f(\boldsymbol{\alpha}) = Q\boldsymbol{\alpha} - \boldsymbol{1} - \frac{1}{t} \begin{bmatrix}
\frac{1}{\alpha_1} \\ \vdots \\ \frac{1}{\alpha_n}
\end{bmatrix}
$$

The Hessian is:

$$
\nabla^2 f(\boldsymbol{\alpha}) = Q + \frac{1}{t} \text{diag}(\alpha_1^{-2}, \dots \ , \alpha_n^{-2})
$$

#### Putting all together in an algorithm: Barrier method
Now we have all the ingridients. Essentially we implement Alogirithm 11.1 from {% cite boyd2004convex %}:

$$
\begin{array}{ll}
\\
\textbf{given} & \text{strictly feasible } \boldsymbol{\alpha}, \quad t := t^{(0)} > 0, \quad \mu > 1, \quad \epsilon > 0 \\
\textbf{repeat} & \\
\quad 1. & \textit{Centering step.} \\
\quad & \quad \text{Compute } \boldsymbol{\alpha}^\star(t) \text{ by minimizing } 
\frac{1}{2} \boldsymbol{\alpha}^\top Q \boldsymbol{\alpha} + \boldsymbol{1}^\top \boldsymbol{\alpha} - \frac{1}{t} \sum_{i=1}^n \log(\alpha_i), \\
\quad & \quad \text{subject to } \mathbf{y}^\top \boldsymbol{\alpha} = 0, \text{ starting at } \boldsymbol{\alpha}. \\
\quad 2. & \textit{Update. } \boldsymbol{\alpha} := \boldsymbol{\alpha}^\star(t). \\
\quad 3. & \textit{Stopping criterion. } \text{quit if } n/t < \epsilon. \\
\quad 4. & \textit{Increase } t. \quad t := \mu t. \\
\end{array}
$$

Eventu


Describe the algorithm, code it, test it

#### MNIST multi-class classification
TODO maybe jsut as an example, but you'll need to use a flexible kernel to fit this without slack. Mostly to introduce SVMs


## SVM
Actually, SVMs are just a small change from what we have here... TODO

#### Wrapping up
Take a look on the literaturte on how SVM are actually optimized, it cannot be that it is something like this, this
implies a nested loop of quadratic programs. I am sure there is a more efficient way to do this. 






<details>
  <summary>Analytical solution dual ascent optimal separating hyperplanes (details)</summary>

Primal problem:

$$
\text{minimize}_{\boldsymbol{\beta}, \beta_0} \quad 
\frac{1}{2} \lVert \boldsymbol{\beta} \rVert^2 
\quad \text{subject to } y_i(\boldsymbol{\beta}^\top  \mathbf{x}_i + \beta_0) \geq 1
$$

$$
\text{Lagrangian:} \quad 
\mathcal{L}(\boldsymbol{\beta}, \beta_0, \boldsymbol{\alpha}) = 
\frac{1}{2} \lVert \boldsymbol{\beta} \rVert^2 
- \sum_{i=1}^n \alpha_i \left[ y_i(\boldsymbol{\beta}^\top  \mathbf{x}_i + \beta_0) - 1 \right]

$$

Step 1: Take derivatives and set to 0 (stationarity)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{\beta}} = \boldsymbol{\beta} - \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i = 0 
\quad \Rightarrow \quad \boldsymbol{\beta} = \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i

$$

$$
\frac{\partial \mathcal{L}}{\partial \beta_0} = -\sum_{i=1}^n \alpha_i y_i = 0 
\quad \Rightarrow \quad \sum_{i=1}^n \alpha_i y_i = 0
$$

Step 2: Plug back into the Lagrangian to get rid of the variables:


$$
\mathcal{L}(\boldsymbol{\alpha}) 
= \frac{1}{2} \left\| \sum_{i=1}^n \alpha_i y_i \mathbf{x}_i \right\|^2 
- \sum_{i=1}^n \alpha_i \left[ y_i \left( \left( \sum_{j=1}^n \alpha_j y_j \mathbf{x}_j \right)^\top  \mathbf{x}_i + \beta_0 \right) - 1 \right]
$$

After simplification and removing terms involving the intercept:

$$
\mathcal{L}(\boldsymbol{\alpha}) 
= \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top  \mathbf{x}_j 
- \sum_{i=1}^n \alpha_i y_i \left( \sum_{j=1}^n \alpha_j y_j \mathbf{x}_j^\top  \mathbf{x}_i \right)
+ \sum_{i=1}^n \alpha_i
= -\frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top  \mathbf{x}_j + \sum_{i=1}^n \alpha_i

$$

Dual problem (maximize the dual function)

$$
\text{maximize}_{\boldsymbol{\alpha}} \quad 
\sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{j=1}^n \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top  \mathbf{x}_j
$$

$$

\text{subject to } \sum_{i=1}^n \alpha_i y_i = 0, \quad \alpha_i \geq 0 \quad \forall i

$$

