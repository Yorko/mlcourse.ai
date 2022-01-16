(topic08_intro)=

# Topic 8. Vowpal Wabbit: Learning with Gigabytes of Data

```{figure} /_static/img/topic8-teaser.png
:name: topic8-teaser
:width: 200px
```

The theoretical part here covert the analysis of Stochastic Gradient Descent, it was this optimization method that made it possible to successfully train both neural networks and linear models on really large training sets. Here we also discuss what can be done in cases of millions of features in a supervised learning task (“hashing trick”) and move on to Vowpal Wabbit, a utility that allows you to train a model with gigabytes of data in a matter of minutes, and sometimes of acceptable quality. We consider several cases including StackOverflow questions tagging with a training set of several gigabytes.

## Steps in this block

1\. Read the [article](topic08) (same in a form of a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/topic-8-online-learning-and-vowpal-wabbit));

2\. Watch a video lecture on coming in 2 parts:
 - [Stochastic Gradient Descent](https://youtu.be/EUSXbdzaQE8);
 - [Vowpal Wabbit](https://www.youtube.com/watch?v=gyCjancgR9U);

3\. Complete [demo assignment 8](assignment08) (same as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/a8-demo-implementing-online-regressor)) which walks you through implementation from scratch, very good for the intuitive understanding of the algorithm;

4\. Check out the [solution](assignment08_solution) (same as a [Kaggle Notebook](https://www.kaggle.com/kashnitsky/a8-demo-implementing-online-regressor-solution)) to the demo assignment (optional);

5\. Complete [Bonus Assignment 8](bonus08) where we go through the math and implement two algorithms -- a regressor and a classifier -- driven by stochastic gradient descent (SGD). (optional, available under Patreon ["Bonus Assignments" tier](https://www.patreon.com/ods_mlcourse)).
