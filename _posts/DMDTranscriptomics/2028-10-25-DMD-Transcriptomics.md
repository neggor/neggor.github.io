---
layout: post
title: Dynamic Mode Decomposition on transcriptomics data.
tag: [Biology, Applied Math]
---
<link href="/css/syntax.css" rel="stylesheet" >
<script type="text/javascript"
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS_CHTML">
</script>
<script type="text/x-mathjax-config">
  MathJax.Hub.Config({
    tex2jax: {
      inlineMath: [['$','$'], ['\\(','\\)']],
      processEscapes: true},
      jax: ["input/TeX","input/MathML","input/AsciiMath","output/CommonHTML"],
      extensions: ["tex2jax.js","mml2jax.js","asciimath2jax.js","MathMenu.js","MathZoom.js","AssistiveMML.js", "[Contrib]/a11y/accessibility-menu.js"],
      TeX: {
      extensions: ["AMSmath.js","AMSsymbols.js","noErrors.js","noUndefined.js"],
      equationNumbers: {
      autoNumber: "AMS"
      }
    }
  });
</script>

## Foreword and references:

The following arises from the idea of merging mechanistic modelling and machine learning in the context of systems biology. The motivation for this study was finding groups of related (coexpressed, corgulated) genes (a.k.a modules) and their dynamcis at the same time, in order to have "time informed" modules. Besides the problems related scarcity of samples, turns out that DMD and Koopman Operator theory are a good fit in this framework.

The following is a summary of DMD literauture and some systems biology literature related to the "modules" idea. The main references are: 

- [The DMD book](http://dmdbook.com/)

- [Data driven science and engineering](https://databookuw.com/)

- [Seminar DMD paper](https://www.annualreviews.org/doi/abs/10.1146/annurev-fluid-030121-015835)

- [Exact DMD](https://arxiv.org/pdf/1312.0041.pdf) (This one provides the framework for using several replicates)

- [Variable projection methods for DMD](https://arxiv.org/pdf/1704.02343.pdf) (for unevenly sampled data) I found the variable projection approach very beautiful, I am lacking more explanation on why the reparameterization here is equivalent to the analytical solution of the ODE though.

- About Koopman Operator theory, I found very interesting and useful: [Koopman Invariant Subspaces and Finite Linear Representations of Nonlinear Dynamical Systems for Control](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0150171), [Deep learning for universal linear embeddings of nonlinear dynamics](https://www.nature.com/articles/s41467-018-07210-0), [Data-Driven Observability Decomposition with Koopman Operators for Optimization of Output Functions of Nonlinear Systems](https://arxiv.org/pdf/2210.09343.pdf) (this one provides a very interesting framework connecting with the phenotype)

- About modules in systems biology: [A review on module detection methods](https://www.nature.com/articles/s41467-018-03424-4), [A review on the concept of modules](https://www.annualreviews.org/doi/full/10.1146/annurev.biophys.36.040306.132725)

- Some papers using ICA in transcriptomics: [ICA for time series](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0181195), [ICA in human data](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1002367) 

  The last paper on ICA has the following plot that ilustrates very well the idea of decomposition methods in trancriptomics:

<p align="center">
  <img src= "/assets/images/DMDTranscriptomics/ICA_example.png" alt="ICA"/>
</p>

  While this is for static data (replicates instead of snapshots in time), it ilustrates nicely the idea of some "hidden processes" (linear combinations of the expression of the genes) the drive the variability of the data. In that case they have individuals, in out case we have time points. The pattern that they derive there are, in our case, the system dynamics.

The code (work in progress) for this study can be found [here](https://github.com/neggor/DMD_transcriptomics).

## Introduction
In this post, I will discuss the application of Dynamic Mode Decomposition (DMD) to transcriptomics data. I will start by giving a brief introduction to transcriptomics data and the dynamical system that govern gene expression. Then I will discuss the DMD algorithm and its variants. Finally, I will present the results of applying DMD to transcriptomics data.

The motivation for this post is to frame gene expression as a dynamical system and experiment with a modern tool for data-driven for system identification, DMD. It arises from the field of fluid dynamics, and it has been used to characterize systems of discretized partial differential equations (which is a very high dimensional kind of data). One of the main points of DMD is that it provides the framework for extracting **temporally coherent structures** from high dimensional time-series data. This is in contraposition with other methods such PCA (POD) or ICA, which are time agnostic.

<p align="center">
  <img src= "/assets/images/DMDTranscriptomics/DMD_layout.png" alt="DMD basic layout"/>
</p>

It works on snapshots of a dynamical process, either from a simulation or from experimental data. In essence, it consist on fitting a linear time invariant ordinary differential equation to the data through regression (OLS). The key point, however, is that it assumes that the data lives in a low dimensional space, and that the dynamics can be recovered in that space. What does this has to do with gene expression?

### Transcriptomics data, modules and dynamical systems.
The connection with gene expression is quite simple, but it has some interesting derivations. Both, in discretized partial differential equations and high-troughput gene-expression data comes in long column vectors, one per snapshot. Both are very high dimensional dynamical systems and both can be modeled in a lower dimensional space. However, the fact that the state of the cell (characterized by its transcriptome) can be reduced to a low dimensional description is a well documented principle (heuristic?) in biology. Often, it is assumed that co-expressed and/or co-regulated genes have a common biological fuction, which allows to conceptualize the state of the plant in the degree of activation of different biological functions (wbiological functions << # genes). 

Gene regulatory networks exhibit a free-scale distribution. Then, it is expected that the dynamics of the system can be approximated reasonably with a low dimensional system that captures those highly connected subgroups. Reduced Order Modelling, then, provides here, not just a computational advantaje (as it can be in physics) but a reasonable inductive bias and regularization strategy to model the dynamics of biological systems.

<p align="center">
  <img src= "/assets/images/DMDTranscriptomics/transcriptomics.png" alt="DMD basic layout"/>
</p>

There is an extense literature in module detection for transcriptomics, a bit scarcer on time series data. DMD in this context can be understood as a way of exploiting the modularity of the data to derive the dynamics of the system.

Now, modelling transcriptomics with a linear model in the sense of:

$$x_{t+1} = \textbf{A}x_t$$

has been done before (add references), but the focus is not so much on the dynamics of the system as it is on inferring a gene regulatory network (gene-gene interaction) which often leads into a "hairy ball problem". The regularization that DMD adds through fitting a model in a low dimensional space and the espectral analysis results in a gene-dynamic relationship (a bipartite graph) instead of an adjacency matrix gene-gene.


### Dynamic Mode Decomposition and its variants.

Well, let's get in to the math! 

First we review PCA/SVD and then DMD, variable selection in DMD and DMD for univenly sampled data. We will focus on the latter, since available rna-seq data is often not uniformly sampled.

We will denote matrices with capital bold letters and vectors with lower case bold letters. We focus on a matrix
 $$\mathbf{X} \in \mathbb{R}^{G\times T}$$ of snapshopts $$\mathbf{x} \in \mathbb{R}^{G}$$ of
the transcriptome extracted at time $$t_i$$ for $$i=1,\dots, T$$, being $$T$$ the total number of
snapshopts and $$G$$ the total number of genes sequenced.

Hence, the matrix $$\mathbf{X}$$ looks like this:

\begin{equation}
    \mathbf{X} = \begin{bmatrix}
        \mathbf{x}_1 & \mathbf{x}_2 & \dots & \mathbf{x}_T.
    \end{bmatrix}
    \label{eq:1}
\end{equation}

Traditionally, PCA consist on first centering the matrix $$\mathbf{X}^T$$ by substracting the mean
per column. Then, the covariance matrix $$\mathbf{C}$$ is computed as:

\begin{equation}
    \mathbf{C} = \frac{1}{T-1}\mathbf{X}\mathbf{X}^T,
\end{equation}

$$\mathbf{C}$$ is a square matrix of size $$G\times G$$ where each element $$C_{ij}$$
is the covariance between the gene $$i$$ and the gene $$j$$. The diagonal elements $$C_{ii}$$ are the variance
of the gene $$i$$.

Then computing the eigenvalue decomposition of $$\mathbf{C}$$. By the Real Spectral Theorem, since $$\mathbf{C}$$
is a real symmetric matrix and hence self-adjoint, the eigenvalues are real and the eigenvectors are orthogonal

\begin{align}
    \mathbf{C} = \mathbf{U}\mathbf{\Lambda}\mathbf{U}^T,
    \label{eq:2}
\end{align}

where $$\mathbf{U}$$ is a matrix of eigenvectors and $$\mathbf{\Lambda}$$ is a diagonal matrix of eigenvalues. Each
eigenvector is associated with an eigenvalue, the bigger the eigenvalue the more variability in the matrix $$\mathbf{X}^T$$
is explained by the corresponding eigenvector. The principal components are then defined as the projection
of the data $$\mathbf{X}^T$$ into the eigenvectors of $$\mathbf{C}$$.

The hope, then, is that the first few eigenvectors convey some biological meaning. In order
to interpret those eigenvectors, the strategy is usually to look at which are the genes that are
most related with that eigenvector. This is usually achieved by
taking the genes with the highest absolute value on that eigenvector given some threshold.This is the same approach that is used for ICA and DMD as we will show. 

It is important to note that the same result can be achieved through Singular Value Decomposition (SVD).
The SVD of $$\mathbf{X}$$ is given by:

\begin{equation}
    \mathbf{X} = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T.
\end{equation}

If the matrix is properly centered the left singular vectors $$\mathbf{U}$$ are the same as the eigenvectors
of \eqref{eq:2}.

It is worth noticing that in the definition of PCA, \eqref{eq:2} would give equivalent results if we 
permuted the columns of $$\mathbf{X}$$. This is because PCA is invariant to permutations in rows or columns. The result is the same multiplied by a permutation matrix. The idea behind DMD is to find a decomposition that exploits the temporal 
relationship of the snapshopts. The approach is then, a perfect fusion of spatial 
dimensionality-reduction methods, such as PCA, with Fourier transforms in time.

The original motiviation is to find a local approximation in the form

\begin{equation}
    \mathbf{x}_{k+1} = \mathbf{A}\mathbf{x}_k,
    \label{eq:5}
\end{equation}
to a complicated system, possibly non-linear, purely from generated or experimental data.

In essence, the algorithm consist on fitting a linear differential equation on the data projected
onto a lower dimensional space. Especifically, it uses Singular Value Decomposition (SVD) to find
the projection matrix $$\mathbf{U}$$ that send the data to that lower dimensional space. Note that,
if the data is appropiately centered, this is similar (equivalent?) to fit a linerar differetial equation on
the principal components. They key point is that the eigenvalues of this linear operator can be
projected back into the original space and hence related to the original dimensions.

In the vanilla version of DMD, we need \ref{eq:1} to be uniformly sampled, this means that $$A$$ in
\ref{eq:5} advances the measurements with the same time step.
Then we construct 
two matrices $$\mathbf{X}_1$$ and $$\mathbf{X}_2$$ defined as.


<p align="center">
  <img src= "https://quicklatex.com/cache3/78/ql_29852541c93941f59b4ddd26bd2a3078_l3.png" alt="An equation"/>
</p>



Importantly, in the case of having several replicates (e.g. biological replicates) of
the same experiment, the snapshots can be concatenated in the same matrix $$\mathbf{X}_1$$ and $$\mathbf{X}_2$$, following the same fashion. E.g. with two replicates, we have two $$\mathbf{X}_1$$, those can be just over the rows to get a longer matrix $$\mathbf{X}_1$$, up to $$\mathbf{X}_{2(T-1)}$$.

In a more general way, we have data pairs $$(\mathbf{x}_{k,j}, \mathbf{x}_{k+1, j})$$
for $$k=1,\dots, T-1$$ and replicate $$j$$, that have to be arranged into that matrix structure such that we can look for the linear operator that minimizes $$||AX_1 - X_2||_F$$ across all replicates for a given treatment.

This regression procedure will yield an unique optimal solution for $$\mathbf{A}$$. Let's see
how this is actually constructed:

First we compute the SVD of $$\mathbf{X}_1$$:

\begin{equation}
    \mathbf{X}_1 = \mathbf{U}\mathbf{\Sigma}\mathbf{V}^T.
    \tag{8}
\end{equation}
    
Now, $$\mathbf{U} \in \mathbb{R}^{G \times T} $$ are our (non-centered) principal components.
We can truncate the matrix $$\mathbf{U}$$ to the first $$r$$ columns. Then we can just
find the $$A$$ of \ref{eq:5} by least squares:
\begin{equation}
    \mathbf{\tilde{\mathbf{A}}} = \mathbf{X}_2\mathbf{V}\mathbf{\Sigma}^{-1}\mathbf{U}^T.
    \label{eq:Ahat}
    \tag{9}
\end{equation}

Note, however that, if we only take the first $$r$$ "components" we will find $$\mathbf{A}$$ 
projected into the lower dimensional space, we call this $$\tilde{\mathbf{A}}$$ to emphasize
that does not need to be $$G\times G$$ but can be $$r\times r$$.

There are two main reasons to truncate the data and actually find the projected $$\mathbf{A}$$.
Basically, regularization and computational efficiency. However, as we mentioned before, in the case of transcriptomics, there is an hypothesis of modularity of the gene regualtory newtork that we aim to exploit using just $$r$$ dimensions instead of $$G$$.

As we mentioned in the introduction, fitting \ref{eq:5} to transcriptomics data has been done before but with a different
objective. These approaches focus on deriving a weighted adjacency matrix from $$\mathbf{A}$$ to infer
gene regulatory networks. While this is feasible for small networks, it is not for high-trhoughput transcriptomics data; 
it is computationally expensive, not parsimonious and it ignores the modular nature of gene-regulatory networks.

In this sense, here we focus on inferring a bipartite graph genes-dynamics $$G \times r$$ instead of a gene-gene adjacency
matrix $$G \times G$$.

We can then eigendecompose $$\mathbf{\tilde{A}}$$:
\begin{equation}
    \mathbf{\tilde{A}}\mathbf{W} = \mathbf{W}\mathbf{\Lambda},
    \tag{10}
\end{equation}

then project them back into the original space:
\begin{equation}
    \mathbf{\Phi} = \mathbf{U}\mathbf{W}.
    \tag{11}
\end{equation}

Where $$\mathbf{\Phi} \in \mathbb{C}^{G \times r}$$ are the DMD modes and the diagonal of $$\mathbf{\Lambda}$$  are the eigenvalues $$ \lambda_i \in \mathbb{C}$$ are the DMD eigenvalues. 
Obtaining $$\mathbf{\Phi}$$ is the goal, the values in those vectors are an indication of the realtionship of a 
gene with the dynamics of the system. The initial conditions or "amplidutes" $$b$$ can be obtained by least squares:
\begin{equation}
    \mathbf{b} = \mathbf{\Phi}^{\dagger}\mathbf{x}_1,
\end{equation}
 where $$\mathbf{\Phi}^{\dagger}$$ is the pseudoinverse of $$\mathbf{\Phi}$$ or just projecting the first snapshot into the eigenbasis of $$\mathbf{\tilde{A}}$$:
\begin{equation}
    \mathbf{b} = \mathbf{W}^{-1}\mathbf{U}^{T}\mathbf{x}_1.
    \tag{12}
\end{equation}

With this we can reconstruct the system as:
\begin{equation}
    \mathbf{x}_k = \mathbf{\Phi}\mathbf{\Lambda}^k\mathbf{b} =
    \mathbf{U}\mathbf{W}\mathbf{\Lambda}^k\mathbf{W}^{-1}\mathbf{U}^{T}\mathbf{x}_1,
    \tag{13}
\end{equation}

which comes from the classical solution of a linear time invariant system (LTI). Note that the initial conditions may as well be calculated from the data, first projecting into the principal components and then into the eigenbasis of $$\mathbf{\tilde{A}}$$.

It is useful to notice that the dynamics of the system can be captured in a Vandermonde matrix:
<p align="center">
  <img src= "https://quicklatex.com/cache3/40/ql_b03d6e6068692ec0e8370b1727369040_l3.png" alt="An equation"/>

</p>


So the whole reconstruction can be written as:

<p align="center">
  <img src= "https://quicklatex.com/cache3/24/ql_f3f02e3dec9913580d77c09e73167e24_l3.png" alt="An equation"/>

</p>

for $$b_1 \dots b_r$$ the entries in $$\mathbf{b}$$.

It is possible to further reduce the dimensionality by deriving $$\mathbf{b}$$ through
lasso type convex optimization procedure instead:
<p align="center">
  <img src= "https://quicklatex.com/cache3/00/ql_226fcb9fea13769897a5355e9cd27e00_l3.png" alt="An equation"/>

</p>

#### Dynamic Mode Decomposition for univenly sampled data.

Unfortunately, transcriptomics experiments are usually not uniformly sampled. This means that
the time step between snapshots is not constant. This is a problem for DMD since for its derivation
through regression we need to have a constant time step.

The main idea is to frame the problem as a non-linear least squares problem. The objective is to parameterize and fit exponential functions
to the data. This, however, comes at a cost. There is not a unique solution and we are not
guaranteed to achieve a global optimum, nevertheless, with a proper initial guess we might find a useful solution.

To see this, first we need to consider the continuous case of \ref{eq:5}:

\begin{equation}
    \frac{d\mathbf{x}}{dt} = \mathbf{\mathit{A}}\mathbf{x},
    \tag{17}
\end{equation}
whose solution is given by:

<p align ="center">
<img src = "https://quicklatex.com/cache3/75/ql_09920a6a3df1b1dc480825d9e1b6dd75_l3.png" alt = "An equation"/>
</p>


To get the whole trajectory at once we can use a matrix like:
<p align ="center">
<img src = "https://quicklatex.com/cache3/d7/ql_94c34452e8b15a687362d2722d843cd7_l3.png" alt = "An equation"/>
</p>

The above matrix can be visualized in the following plot, where each row has a color:
<p align ="center">
<img src = "/assets/images/DMDTranscriptomics/dynamics_example.png" alt = "An equation"/>
</p>

Importantly, note that $t_i$ now can be an arbitrary time. The idea is to reframe the problem into:

\begin{equation}
    \mathbf{X} = \mathbf{B}\Omega(\mathbf{\lambda}, t),
    \tag{20}
\end{equation}

where $\mathbf{\Omega}(\alpha, t)$ is a matrix of exponentials, where every row is the evaluation of an exponential function in different time points (not necessarily ordered, not necessarily unique). These functions are paremeterized by the eigenvalues $\lambda_i \in \mathbb{C}$.$\mathbf{B} \in \mathbb{C}^{G \times r}$ is the matrix of eigenvectors. Then
$b_i = ||\mathbf{U}\mathbf{B}(:, i)||_2$ and our modes are:

<p align ="center">
<img src = "https://quicklatex.com/cache3/8f/ql_1b9cda415bcb8d8c5b709319b3007b8f_l3.png" alt = "An equation"/>
</p>

It turns out that this problem can be solved in a very efficient and beautiful way using a variable
projection algorithm. This consists on alternating between
solving for $\mathbf{\lambda}$ and $\mathbf{B}$ until convergence. For a sensible initial
choice of $\mathbf{\lambda}$, $\mathbf{B}$ can be found via least squares (using the pseudoinverse $$\dagger$$) as:
\begin{equation}
    \mathbf{B} = \mathbf{X}\Omega(\mathbf{\lambda}, t)^{\dagger},
    \tag{22}
\end{equation}
then improve $\mathbf{\lambda}$ by solving a non-linear least squares problem:
\begin{equation}
    \mathbf{\lambda} = \underset{\mathbf{\lambda}}{\mathrm{argmin}} ||\mathbf{X} - \mathbf{B}\Omega(\mathbf{\lambda}, t)||_F. 
    \tag{23}
\end{equation}
and repeat until convergence. In this case, I use the trust-region-reflective algorithm implemented in scipy in order to enforce an stable system (eigenvalues with negative real part). All this boils down to finding some exponential curves that fit the data in the principal components space, and the relationship of the curves with the principal components. When projected back to the original (gene) space, the result is a set of curves/dynamics that coarsely fit the data and the relationship of every gene with those dynamics.

The initial value suggestion is obtained from a DMD like algorithm (or in the case of evenly sampled data from vanilla DMD itself). In this case we use algorithm 4 from [here](https://arxiv.org/pdf/1704.02343.pdf).

## Experiments
Since transcriptomics data sets usually perform a couple of related experiments I use the whole dataset to generate the projection $$\mathbf{U}$$ and then I use the replicates of each experiment to find the eigenvalues and eigenvectors.

In order to assess the relevance of the experiment, I use the following procedure:
TODO

## Analysis of results
TODO

## Some comments on Koopman Operator theory in Systems Biology