---
layout: post
title: Support Vector Machines
tag: ML
---
<link href="/css/syntax.css" rel="stylesheet" >
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

# From Optimal Separating Hyperplane to Support Vector Machines:

This continues the previous post on [Optimal Separating Hyperplane](Understanding-SVMs_1.html).

The optimal separating hyperplane (OSH) will fail (TODO show this) if the data is not perfectly linearly seaprable. Arguably, we can always find a basis in which the data is linearly separable, but this could lead to overfitting issues.

Support Vector Machines (SVMs) address this problem by allowing for an "slack", a budget of how many points and how much they can be on the wrong side of the hyperplane. 

Following Elements Of Statitical Learning (ESL), the idea is to allow for error in the following way:

$$y_i(x^T\beta + \beta_0) \geq M(1 - \xi) = M - M\xi$$ 


$$M$$ is the margin, $$\xi > 1$$ accounts for wrong calssifications, $$0 < \xi < 1$$ for observations inside the margin. From the OSH constraint $$y_i(x^T\beta + \beta_0) \geq 1$$, we can rewrite easily add the "slack":

$$y_i(x^T\beta + \beta_0) \geq 1 - \xi_i$$

So the optimization problem becomes:

$$\min_{\beta, \beta_0,} \frac{1}{2}\lVert\beta\rVert^2 + C\sum_{i=1}^N \xi_i$$


$$\text{s.t.} \quad y_i(x^T\beta + \beta_0) \geq 1 - \xi_i, \xi \geq 0 \quad \forall \quad i = 1, \dots, n$$

C is just a hyperparameter, a regularization constant. High values of C imply a huge cost of having error, which need to be compensated increasing the norm of $$\beta$$. Hence, there is a trade-off between the error and the norm of $$\beta$$. Therefore, high values of C imply no regularization, and possibly overfitting. On the other hand, low values of C "ignore" the error in the training data and focus on maximizing the margin. It is a beautiful way to frame bias-variance trade-off.

Now, as with the OSH, we construct the Lagrangian as:

$$Lp = \frac{1}{2}\lVert\beta\rVert^2 + C\sum_{i=1}^N\xi_i - \sum^N_{i=1}\alpha_i[y_i(x_i^T\beta + \beta_0)-(1-\xi_i)] - \sum_{i=1}^N\mu_i\xi_i$$

The signs indicate that if the constraint is violated there is a penalty. Taking the derivatives and equating to 0 is quite straightforward having the previous result from OSH:

$$\beta = \sum_{i=0}^N \alpha_i y_i x_i$$

$$\sum_{i=1}^N \alpha_iy_i=0$$

$$\alpha_i = C - \mu_i$$


This, essentially, yields the same maximization problem as with OSH:

$$ \max_{\mathbf{\alpha}} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i=1}^n \sum_{k=1}^n \alpha_i \alpha_j y_i y_k \mathbf{x}_i^T \mathbf{x}_k$$

$$\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^N \alpha_iy_i=0$$

The only difference being that $$\alpha_8$$ is bounded by C. In fact, in terms of code, the only change needed to go from OSH to SVM is, from these bounds:
```python
my_bounds = [(0, np.inf)] * len(y)
```
to these, for some C:
```python
my_bounds = [(0, C)] * len(y)
```
I found it quite surprising that this small change leads to such a different behavior. In a way, OSH implies no regularization at all, so C is set to infinity. That would set OSH as a particular case of SVMs.

TODO: Study time: get again over the meaining of the lagrange multipliers, what happens if 
you limit it? Sounds like you are limiting the penalty that you are willing to pay for
misclassification. I think it is indeed boiling down to the penalty that you do. But to 
understand this properly I need to understand the duality thing better. 

Indeed, if we recall our intuotuion on the geomtry of this thing. If the margin is decided by the sensibility to movements in x, limiting the alphas (essentially limiting the betas) is limiting the sesibility to x, hence, increasing the margin. That is actually the main point... Now the thing is how this relates to the error and the slack!

The thing is that in the previous blogpost I navigated not knowing of this stuff in depth kind of succesfully... now this is not going to happen. I need to understand the lagrange multipliers properly, because I am essentially putting a constraint on them!

When they do the minimization step the idea is that the alphas go to infinity. But here because th epenalty is limited, it allows to further misscassify... And also it is limiting the effect of those suport vectors in beta...

TODO visualization of the effects of C.

# Revisiting Lagrange Multipliers
# Kernels

TODO
Reflection on kernels, curse of dimensionality and so on... lay the ground to talk about
transformers, linear attention and so on...