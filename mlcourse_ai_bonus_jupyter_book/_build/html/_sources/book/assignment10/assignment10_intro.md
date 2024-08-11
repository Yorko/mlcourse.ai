(assignment10_intro)=

# Topic 10. Gradient Boosting 

<div align="center">
<img src="../../_static/img/topic10-teaser.jpeg" /> 
</div>
<br> 

Gradient boosting is one of the most prominent Machine Learning algorithms, it founds a lot of industrial applications. For instance, the Yandex search engine is a big and complex system with gradient boosting (MatrixNet) somewhere deep inside. Many recommender systems are also built on boosting. It is a very versatile approach applicable to classification, regression, and ranking. Therefore, here we cover both theoretical basics of gradient boosting and specifics of most widespread implementations – Xgboost, LightGBM, and Catboost.

1\. Read the [article](https://mlcourse.ai/articles/topic10-boosting/) (same in a form of a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-10-gradient-boosting))

2\. Watch a video lecture coming in 2 parts:

 - [part 1](https://youtu.be/g0ZOtzZqdqk), fundamental ideas behind gradient boosting
 - [part 2](https://youtu.be/V5158Oug4W8), key ideas behind major implementations: Xgboost, LightGBM, and CatBoost
 
3\. **Kaggle:** Take a look at the "Flight delays" [competition](https://www.kaggle.com/c/flight-delays-fall-2018) and a starter with [CatBoost](https://www.kaggle.com/kashnitsky/mlcourse-ai-fall-2019-catboost-starter-with-gpu). Start analyzing data and building features to improve your solution. Try to improve the best publicly shared solution by at least 0.5%. But still, please do not share high-performing solutions, it ruins the competitive spirit.
 
## Bonus Assignment 10. Implementation of the gradient boosting algorithm

In this assignment, we go through the math and implement the general gradient boosting algorithm - the same class will implement a binary classifier that minimizes the logistic loss function and two regressors that minimize the mean squared error (MSE) and the root mean squared logarithmic error (RMSLE). This way, we will see that we can optimize arbitrary differentiable functions using gradient boosting and how this technique adapts to different contexts.

<div align="center">
<img src="../../_static/img/assignment10_teaser_math.png" /> 
</div>
<br>

Residuals at each gradient boosting iteration and the corresponding tree prediction:

<div align="center">
<img src="../../_static/img/assignment10_teaser_residuals.png" /> 
</div>
<br>