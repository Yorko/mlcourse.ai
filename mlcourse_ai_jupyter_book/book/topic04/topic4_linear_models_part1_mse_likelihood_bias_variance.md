---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(topic04_part1)=


# Topic 4. Linear Classification and Regression
## Part 1. Ordinary Least Squares

```{figure} /_static/img/ods_stickers.jpg
```

Author: [Pavel Nesterov](http://pavelnesterov.info/). Translated and edited by [Christina Butsko](https://www.linkedin.com/in/christinabutsko/), [Yury Kashnitsky](https://yorko.github.io), [Nerses Bagiyan](https://www.linkedin.com/in/nersesbagiyan/), [Yulia Klimushina](https://www.linkedin.com/in/yuliya-klimushina-7168a9139), and [Yuanyuan Pao](https://www.linkedin.com/in/yuanyuanpao/). This material is subject to the terms and conditions of the [Creative Commons CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Free use is permitted for any non-commercial purpose.



## Article outline
1. [Introduction](#introduction)
2. [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
3. [Bias-Variance Decomposition](#bias-variance-decomposition)
4. [Regularization of Linear Regression](#regularization-of-linear-regression)
5. [Useful resources](#useful-resources)


## 1. Introduction
We will start studying linear models with linear regression. First of all, you must specify a model that relates the dependent variable $y$ to the explanatory factors (or features); for linear models, the dependency function will take the following form: $\large y = w_0 + \sum_{i=1}^m w_i x_i$, where $m$ is the number of features. If we add a fictitious dimension $x_0 = 1$ (called _bias_ or _intercept_ term) for each observation, then the linear form can be rewritten in a slightly more compact way by pulling the absolute term $w_0$ into the sum: $\large y = \sum_{i=0}^m w_i x_i = \textbf{w}^\text{T} \textbf{x}$. If we have a matrix of $n$ observations, where the rows are observations from a data set, we need to add a single column of ones on the left. We define the model as follows:

$$\large \textbf y = \textbf X \textbf w + \epsilon,$$

where

- $\textbf w \in \mathbb{R}^{m+1}$ – is a $(m+1) \times 1$ column-vector of the model parameters (in machine learning, these parameters are often referred to as *weights*);
- $\textbf X \in \mathbb{R}^{n \times (m+1)}$ – is a $n \times (m+1)$ matrix of observations and their features, (including the fictitious column on the left) with full column [rank](https://en.wikipedia.org/wiki/Rank_(linear_algebra)):  $\text{rank}\left(\textbf X\right) = m + 1 $;
- $\epsilon \in \mathbb{R}^n$ – is a $n \times 1$ random column-vector, referred to as *error* or *noise*;
- $\textbf y \in \mathbb{R}^n$ – is a $n \times 1$ column-vector - the dependent (or *target*) variable.

We can also write this expression out for each observation

$$\large y_i = \sum_{j=0}^m w_j X_{ij} + \epsilon_i$$

Will apply the following restrictions to the set of random errors $\epsilon_i$ (see [Gauss-Markov](https://en.wikipedia.org/wiki/Gauss%E2%80%93Markov_theorem) theorem for deeper statistical motivation):

- expectation of all random errors is zero:  $\forall i: \mathbb{E}\left[\epsilon_i\right] = 0 $;
- all random errors have the same finite variance, this property is called <a href="https://en.wikipedia.org/wiki/Homoscedasticity">homoscedasticity</a>:  $\forall i: \text{Var}\left(\epsilon_i\right) = \sigma^2 < \infty $;
- random errors are uncorrelated:  $\forall i \neq j: \text{Cov}\left(\epsilon_i, \epsilon_j\right) = 0 $.

[Estimator](https://en.wikipedia.org/wiki/Estimator)  $\widehat{w}_i $ of weights  $w_i $ is called *linear* if

$$\large \widehat{w}_i = \omega_{1i}y_1 + \omega_{2i}y_2 + \cdots + \omega_{ni}y_n,$$

where  $\forall\ k\ $, $\omega_{ki} $
depend only on the samples from $\textbf X$, but not on $\textbf w$ - we don't know the true coefficients $w_i$ (that's why we try to estimate them with the given data). Since the solution for finding the optimal weights is a linear estimator, the model is called *linear regression*.

**Disclaimer:** at this point, it's time to say that we don't plan to squeeze a whole statistics course into one article, so we try to provide the general idea of the math behind OLS.

Let's introduce one more definition. Estimator $\widehat{w}_i $ is called *unbiased* when its expectation is equal to the real but unknown value of the estimated parameter:

$$\Large \mathbb{E}\left[\widehat{w}_i\right] = w_i$$

One of the ways to calculate those weights is with the ordinary least squares method (OLS), which minimizes the mean squared error between the actual value of the dependent variable and the predicted value given by the model:

$$\Large \begin{array}{rcl}\mathcal{L}\left(\textbf X, \textbf{y}, \textbf{w} \right) &=& \frac{1}{2n} \sum_{i=1}^n \left(y_i - \textbf{w}^\text{T} \textbf{x}_i\right)^2 \\
&=& \frac{1}{2n} \left\| \textbf{y} - \textbf X \textbf{w} \right\|_2^2 \\
&=& \frac{1}{2n} \left(\textbf{y} - \textbf X \textbf{w}\right)^\text{T} \left(\textbf{y} - \textbf X \textbf{w}\right)
\end{array}$$

To solve this optimization problem, we need to calculate derivatives with respect to the model parameters. We set them to zero and solve the resulting equation for $\textbf w$ (matrix differentiation may seem difficult; try to do it in terms of sums to be sure of the answer).


<br>
<details>
<summary>Small CheatSheet on matrix derivatives (click the triangle to extend)</summary>
<p>

$$\large \begin{array}{rcl}
\frac{\partial}{\partial \textbf{X}} \textbf{X}^{\text{T}} \textbf{A} &=& \textbf{A} \\
\frac{\partial}{\partial \textbf{X}} \textbf{X}^{\text{T}} \textbf{A} \textbf{X} &=& \left(\textbf{A} + \textbf{A}^{\text{T}}\right)\textbf{X} \\
\frac{\partial}{\partial \textbf{A}} \textbf{X}^{\text{T}} \textbf{A} \textbf{y} &=&  \textbf{X}^{\text{T}} \textbf{y}\\
\frac{\partial}{\partial \textbf{X}} \textbf{A}^{-1} &=& -\textbf{A}^{-1} \frac{\partial \textbf{A}}{\partial \textbf{X}} \textbf{A}^{-1}
\end{array}$$

</p>
</details>


What we get is:

$$\Large \begin{array}{rcl} \frac{\partial \mathcal{L}}{\partial \textbf{w}} &=& \frac{\partial}{\partial \textbf{w}} \frac{1}{2n} \left( \textbf{y}^{\text{T}} \textbf{y} -2\textbf{y}^{\text{T}} \textbf{X} \textbf{w} + \textbf{w}^{\text{T}} \textbf{X}^{\text{T}} \textbf{X} \textbf{w}\right) \\
&=& \frac{1}{2n} \left(-2 \textbf{X}^{\text{T}} \textbf{y} + 2\textbf{X}^{\text{T}} \textbf{X} \textbf{w}\right)
\end{array}$$

$$\Large \begin{array}{rcl} \frac{\partial \mathcal{L}}{\partial \textbf{w}} = 0 &\Leftrightarrow& \frac{1}{2n} \left(-2 \textbf{X}^{\text{T}} \textbf{y} + 2\textbf{X}^{\text{T}} \textbf{X} \textbf{w}\right) = 0 \\
&\Leftrightarrow& -\textbf{X}^{\text{T}} \textbf{y} + \textbf{X}^{\text{T}} \textbf{X} \textbf{w} = 0 \\
&\Leftrightarrow& \textbf{X}^{\text{T}} \textbf{X} \textbf{w} = \textbf{X}^{\text{T}} \textbf{y} \\
&\Leftrightarrow& \textbf{w} = \left(\textbf{X}^{\text{T}} \textbf{X}\right)^{-1} \textbf{X}^{\text{T}} \textbf{y}
\end{array}$$

Bearing in mind all the definitions and conditions described above, we can say that, based on the <a href="https://en.wikipedia.org/wiki/Gauss–Markov_theorem">Gauss–Markov theorem</a>, OLS estimators of the model parameters are optimal among all linear and unbiased estimators, i.e. they give the lowest variance.

## 2. Maximum Likelihood Estimation

One could ask why we choose to minimize the mean square error instead of something else? After all, one can minimize the mean absolute value of the residual. The only thing that will happen, if we change the minimized value, is that we will exceed the Gauss-Markov theorem conditions, and our estimates will therefore cease to be the optimal over the linear and unbiased ones.

Before we continue, let's digress to illustrate maximum likelihood estimation with a simple example.

Many people probably remember the formula of ethyl alcohol, so I decided to do an experiment to determine whether people remember a simpler formula for methanol: $CH_3OH$. We surveyed 400 people to find that only 117 people remembered the formula. Then, it is reasonable to assume that the probability that the next respondent knows the formula of methyl alcohol is $\frac{117}{400} \approx 29\%$. Let's show that this intuitive assessment is not only good but also a maximum likelihood estimate. Where this estimate come from? Recall the definition of the Bernoulli distribution: a random variable has a <a href="https://en.wikipedia.org/wiki/Bernoulli_distribution">Bernoulli distribution</a> if it takes only two values ($1$ and $0$ with probability $\theta$ and $1 - \theta$, respectively) and has the following probability distribution function:

$$\Large p\left(\theta, x\right) = \theta^x \left(1 - \theta\right)^\left(1 - x\right), x \in \left\{0, 1\right\}$$

This distribution is exactly what we need, and the distribution parameter $\theta$ is the estimate of the probability that a person knows the formula of methyl alcohol. In our $400$ *independent* experiments, let's denote their outcomes as $\textbf{x} = \left(x_1, x_2, \ldots, x_{400}\right)$. Let's write down the likelihood of our data (observations), i.e. the probability of observing exactly these 117 realizations of the random variable $x = 1$ and 283 realizations of $x = 0$:

$$\Large p(\textbf{x}; \theta) = \prod_{i=1}^{400} \theta^{x_i} \left(1 - \theta\right)^{\left(1 - x_i\right)} = \theta^{117} \left(1 - \theta\right)^{283}$$

Next, we will maximize the expression with respect to $\theta$. Most often this is not done with the likelihood $p(\textbf{x}; \theta)$ but with its logarithm (the monotonic transformation does not change the solution but simplifies calculation greatly):

$$\Large \log p(\textbf{x}; \theta) = \log \prod_{i=1}^{400} \theta^{x_i} \left(1 - \theta\right)^{\left(1 - x_i\right)} = $$
$$ \large = \log \theta^{117} \left(1 - \theta\right)^{283} =  117 \log \theta + 283 \log \left(1 - \theta\right)$$

Now, we want to find such a value of $\theta$ that will maximize the likelihood. For this purpose, we'll take the derivative with respect to $\theta$, set it to zero, and solve the resulting equation:

$$\Large  \frac{\partial \log p(\textbf{x}; \theta)}{\partial \theta} = \frac{\partial}{\partial \theta} \left(117 \log \theta + 283 \log \left(1 - \theta\right)\right) = \frac{117}{\theta} - \frac{283}{1 - \theta};$$

It turns out that our intuitive assessment is exactly the maximum likelihood estimate. Now let us apply the same reasoning to the linear regression problem and try to find out what lies beyond the mean squared error. To do this, we'll need to look at linear regression from a probabilistic perspective. Our model naturally remains the same:

$$\Large \textbf y = \textbf X \textbf w + \epsilon,$$

but let us now assume that the random errors follow a centered [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) (beware, it's an additional assumption, it's not a prerequisite of a Gauss-Markov theorem):

$$\Large \epsilon_i \sim \mathcal{N}\left(0, \sigma^2\right)$$

Let's rewrite the model from a new perspective:

$$\Large \begin{array}{rcl}
y_i &=& \sum_{j=1}^m w_j X_{ij} + \epsilon_i \\
&\sim& \sum_{j=1}^m w_j X_{ij} + \mathcal{N}\left(0, \sigma^2\right) \\
p\left(y_i \mid \textbf X; \textbf{w}\right) &=& \mathcal{N}\left(\sum_{j=1}^m w_j X_{ij}, \sigma^2\right)
\end{array}$$

Since the examples are taken independently (uncorrelated errors is one of the conditions of the Gauss-Markov theorem), the full likelihood of the data will look like a product of the density functions $p\left(y_i\right)$. Let's consider the log-likelihood, which allows us to switch products to sums:

$$\Large \begin{array}{rcl}
\log p\left(\textbf{y}\mid \textbf X; \textbf{w}\right) &=& \log \prod_{i=1}^n \mathcal{N}\left(\sum_{j=1}^m w_j X_{ij}, \sigma^2\right) \\
&=& \sum_{i=1}^n \log \mathcal{N}\left(\sum_{j=1}^m w_j X_{ij}, \sigma^2\right) \\
&=& -\frac{n}{2}\log 2\pi\sigma^2 -\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i - \textbf{w}^\text{T} \textbf{x}_i\right)^2
\end{array}$$

We want to find the maximum likelihood hypothesis i.e. we need to maximize the expression $p\left(\textbf{y} \mid \textbf X; \textbf{w}\right)$ to get $\textbf{w}_{\text{ML}}$, which is the same as maximizing its logarithm. Note that, while maximizing the function over some parameter, you can throw away all the members that do not depend on this parameter:

$$\Large \begin{array}{rcl}
\textbf{w}_{\text{ML}} &=& \arg \max_{\textbf w} p\left(\textbf{y}\mid \textbf X; \textbf{w}\right) = \arg \max_{\textbf w} \log p\left(\textbf{y}\mid \textbf X; \textbf{w}\right)\\
&=& \arg \max_{\textbf w} -\frac{n}{2}\log 2\pi\sigma^2 -\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i - \textbf{w}^{\text{T}} \textbf{x}_i\right)^2 \\
&=& \arg \max_{\textbf w} -\frac{1}{2\sigma^2} \sum_{i=1}^n \left(y_i - \textbf{w}^{\text{T}} \textbf{x}_i\right)^2 \\
&=&  \arg \min_{\textbf w} \mathcal{L}\left(\textbf X, \textbf{y}, \textbf{w} \right)
\end{array}$$

Thus, we have seen that the maximization of the likelihood of data is the same as the minimization of the mean squared error (given the above assumptions). It turns out that such a cost function is a consequence of the fact that the errors are distributed normally.


## 3. Bias-Variance Decomposition

Let's talk a little about the error properties of linear regression prediction (in fact, this discussion is valid for all machine learning algorithms). We just covered the following:

- true value of the target variable is the sum of a deterministic function $f\left(\textbf{x}\right)$ and random error $\epsilon$: $y = f\left(\textbf{x}\right) + \epsilon$;
- error is normally distributed with zero mean and some variance: $\epsilon \sim \mathcal{N}\left(0, \sigma^2\right)$;
- true value of the target variable is also normally distributed: $y \sim \mathcal{N}\left(f\left(\textbf{x}\right), \sigma^2\right)$;
- we try to approximate a deterministic but unknown function $f\left(\textbf{x}\right)$ using a linear function of the covariates $\widehat{f}\left(\textbf{x}\right)$, which, in turn, is a point estimate of the function $f$ in function space (specifically, the family of linear functions that we have limited our space to), i.e. a random variable that has mean and variance.

So, the error at the point $\textbf{x}$ decomposes as follows:

$$\Large \begin{array}{rcl}
\text{Err}\left(\textbf{x}\right) &=& \mathbb{E}\left[\left(y - \widehat{f}\left(\textbf{x}\right)\right)^2\right] \\
&=& \mathbb{E}\left[y^2\right] + \mathbb{E}\left[\left(\widehat{f}\left(\textbf{x}\right)\right)^2\right] - 2\mathbb{E}\left[y\widehat{f}\left(\textbf{x}\right)\right] \\
&=& \mathbb{E}\left[y^2\right] + \mathbb{E}\left[\widehat{f}^2\right] - 2\mathbb{E}\left[y\widehat{f}\right] \\
\end{array}$$

For clarity, we will omit the notation of the argument of the functions. Let's consider each member separately. The first two are easily decomposed according to the formula $\text{Var}\left(z\right) = \mathbb{E}\left[z^2\right] - \mathbb{E}\left[z\right]^2$:

$$\Large \begin{array}{rcl}
\mathbb{E}\left[y^2\right] &=& \text{Var}\left(y\right) + \mathbb{E}\left[y\right]^2 = \sigma^2 + f^2\\
\mathbb{E}\left[\widehat{f}^2\right] &=& \text{Var}\left(\widehat{f}\right) + \mathbb{E}\left[\widehat{f}\right]^2 \\
\end{array}$$

Note that:

$$\Large \begin{array}{rcl}
\text{Var}\left(y\right) &=& \mathbb{E}\left[\left(y - \mathbb{E}\left[y\right]\right)^2\right] \\
&=& \mathbb{E}\left[\left(y - f\right)^2\right] \\
&=& \mathbb{E}\left[\left(f + \epsilon - f\right)^2\right] \\
&=& \mathbb{E}\left[\epsilon^2\right] = \sigma^2
\end{array}$$

$$\Large \mathbb{E}[y] = \mathbb{E}[f + \epsilon] = \mathbb{E}[f] + \mathbb{E}[\epsilon] = f$$

And finally, we get to the last term in the sum. Recall that the error and the target variable are independent of each other:

$$\Large \begin{array}{rcl}
\mathbb{E}\left[y\widehat{f}\right] &=& \mathbb{E}\left[\left(f + \epsilon\right)\widehat{f}\right] \\
&=& \mathbb{E}\left[f\widehat{f}\right] + \mathbb{E}\left[\epsilon\widehat{f}\right] \\
&=& f\mathbb{E}\left[\widehat{f}\right] + \mathbb{E}\left[\epsilon\right] \mathbb{E}\left[\widehat{f}\right]  = f\mathbb{E}\left[\widehat{f}\right]
\end{array}$$

Finally, let's bring this all together:

$$\Large \begin{array}{rcl}
\text{Err}\left(\textbf{x}\right) &=& \mathbb{E}\left[\left(y - \widehat{f}\left(\textbf{x}\right)\right)^2\right] \\
&=& \sigma^2 + f^2 + \text{Var}\left(\widehat{f}\right) + \mathbb{E}\left[\widehat{f}\right]^2 - 2f\mathbb{E}\left[\widehat{f}\right] \\
&=& \left(f - \mathbb{E}\left[\widehat{f}\right]\right)^2 + \text{Var}\left(\widehat{f}\right) + \sigma^2 \\
&=& \text{Bias}\left(\widehat{f}\right)^2 + \text{Var}\left(\widehat{f}\right) + \sigma^2
\end{array}$$

With that, we have reached our ultimate goal -- the last formula tells us that the forecast error of any model of type $y = f\left(\textbf{x}\right) + \epsilon$ is composed of:

- squared bias: $\text{Bias}\left(\widehat{f}\right)$ is the average error for all sets of data;
- variance: $\text{Var}\left(\widehat{f}\right)$ is error variability, or by how much error will vary if we train the model on different sets of data;
- irremovable error: $\sigma^2$.

While we cannot do anything with the $\sigma^2$ term, we can influence the first two. Ideally, we'd like to negate both of these terms (upper left square of the picture), but, in practice, it is often necessary to balance between the biased and unstable estimates (high variance).

```{figure} /_static/img/topic4_bias_variance.png
:width: 480px
```

Generally, as the model becomes more computational (e.g. when the number of free parameters grows), the variance (dispersion) of the estimate also increases, but bias decreases. Due to the fact that the training set is memorized completely instead of generalizing, small changes lead to unexpected results (overfitting). On the other side, if the model is too weak, it will not be able to learn the pattern, resulting in learning something different that is offset with respect to the right solution.

```{figure} /_static/img/topic4_bias_variance2.png
:width: 480px
```

The Gauss-Markov theorem asserts that the OLS estimator of parameters of the linear model is the best for the class of linear unbiased estimator. This means that if there exists any other unbiased model $g$, from the same class of linear models, we can be sure that $Var\left(\widehat{f}\right) \leq Var\left(g\right)$.

## 4. Regularization of Linear Regression

There are situations where we might intentionally increase the bias of the model for the sake of stability i.e. to reduce the variance of the model $\text{Var}\left(\widehat{f}\right)$. One of the conditions of the Gauss-Markov theorem is the full column rank of matrix $\textbf{X}$. Otherwise, the OLS solution $\textbf{w} = \left(\textbf{X}^\text{T} \textbf{X}\right)^{-1} \textbf{X}^\text{T} \textbf{y}$ does not exist since the inverse matrix $\left(\textbf{X}^\text{T} \textbf{X}\right)^{-1}$ does not exist. In other words, matrix $\textbf{X}^\text{T} \textbf{X}$ will be singular or degenerate. This problem is called an <a href="https://en.wikipedia.org/wiki/Well-posed_problem"> ill-posed problem</a>. Problems like this must be corrected, namely, matrix $\textbf{X}^\text{T} \textbf{X}$ needs to become non-degenerate, or regular (which is why this process is called regularization). Often we observe the so-called multicollinearity in the data: when two or more features are strongly correlated, it is manifested in the matrix $\textbf{X}$ in the form of "almost" linear dependence between the columns. For example, in the problem of predicting house prices by their parameters, attributes "area with balcony" and "area without balcony" will have an "almost" linear relationship. Formally, matrix $\textbf{X}^\text{T} \textbf{X}$ for such data is reversible, but, due to multicollinearity, some of its eigenvalues will be close to zero. In the inverse matrix $\textbf{X}^\text{T} \textbf{X}$, some extremely large eigenvalues will appear, as eigenvalues of the inverse matrix are $\frac{1}{\lambda_i}$. The result of this vacillation of eigenvalues is an unstable estimate of model parameters, i.e. adding a new set of observations to the training data will lead to a completely different solution.
One method of regularization is <a href="https://en.wikipedia.org/wiki/Tikhonov_regularization">Tikhonov regularization</a>, which generally looks like the addition of a new member to the mean squared error:

$$\Large \begin{array}{rcl}
\mathcal{L}\left(\textbf{X}, \textbf{y}, \textbf{w} \right) &=& \frac{1}{2n} \left\| \textbf{y} - \textbf{X} \textbf{w} \right\|_2^2 + \left\| \Gamma \textbf{w}\right\|^2\\
\end{array}$$

The Tikhonov matrix is often expressed as the product of a number by the identity matrix: $\Gamma = \frac{\lambda}{2} E$. In this case, the problem of minimizing the mean squared error becomes a problem with a restriction on the $L_2$ norm. If we differentiate the new cost function with respect to the model parameters, set the resulting function to zero, and rearrange for  $\textbf{w}$, we get the exact solution of the problem.

$$\Large \begin{array}{rcl}
\textbf{w} &=& \left(\textbf{X}^{\text{T}} \textbf{X} + \lambda \textbf{E}\right)^{-1} \textbf{X}^{\text{T}} \textbf{y}
\end{array}$$

This type of regression is called ridge regression. The ridge is the diagonal matrix that we add to the $\textbf{X}^\text{T} \textbf{X}$ matrix to ensure that we get a regular matrix as a result.

```{figure} /_static/img/topic4_ridge.png
:width: 480px
```

Such a solution reduces dispersion but becomes biased because the norm of the vector of parameters is also minimized, which makes the solution shift towards zero. On the figure below, the OLS solution is at the intersection of the white dotted lines. Blue dots represent different solutions of ridge regression. It can be seen that by increasing the regularization parameter $\lambda$, we shift the solution towards zero.

```{figure} /_static/img/topic4_l2_regul.png
:width: 480px
```
## 5. Useful resources
- Medium ["story"](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-4-linear-classification-and-regression-44a41b9b5220) based on this notebook
- Main course [site](https://mlcourse.ai), [course repo](https://github.com/Yorko/mlcourse.ai), and YouTube [channel](https://www.youtube.com/watch?v=QKTuw4PNOsU&list=PLVlY_7IJCMJeRfZ68eVfEcu-UcN9BbwiX)
- Course materials as a [Kaggle Dataset](https://www.kaggle.com/kashnitsky/mlcourse)
- If you read Russian: an [article](https://habrahabr.ru/company/ods/blog/323890/) on Habrahabr with ~ the same material. And a [lecture](https://youtu.be/oTXGQ-_oqvI) on YouTube
- A nice and concise overview of linear models is given in the book ["Deep Learning"](http://www.deeplearningbook.org) (I. Goodfellow, Y. Bengio, and A. Courville).
- Linear models are covered practically in every ML book. We recommend "Pattern Recognition and Machine Learning" (C. Bishop) and "Machine Learning: A Probabilistic Perspective" (K. Murphy).
- If you prefer a thorough overview of linear model from a statistician's viewpoint, then look at "The elements of statistical learning" (T. Hastie, R. Tibshirani, and J. Friedman).
- The book "Machine Learning in Action" (P. Harrington) will walk you through implementations of classic ML algorithms in pure Python.
- [Scikit-learn](http://scikit-learn.org/stable/documentation.html) library. These guys work hard on writing really clear documentation.
- Scipy 2017 [scikit-learn tutorial](https://github.com/amueller/scipy-2017-sklearn) by Alex Gramfort and Andreas Mueller.
- One more [ML course](https://github.com/diefimov/MTH594_MachineLearning) with very good materials.
- [Implementations](https://github.com/rushter/MLAlgorithms) of many ML algorithms. Search for linear regression and logistic regression.
