(bonus03)=

# Bonus Assignment 3. Decision trees

```{figure} /_static/img/topic3-teaser.png
:width: 200px
```

You can purchase a Bonus Assignments pack with the best non-demo versions of [mlcourse.ai](https://mlcourse.ai/) assignments. Select the ["Bonus Assignments" tier](https://www.patreon.com/ods_mlcourse) on Patreon or a [similar tier](https://boosty.to/ods_mlcourse/purchase/1142055?ssource=DIRECT&share=subscription_link) on Boosty (rus).

<div class="row">
  <div class="col-md-8" markdown="1">
  <p align="center">
  <a href="https://www.patreon.com/ods_mlcourse">
         <img src="../../_static/img/become_a_patron.png">
  </a>
  &nbsp;&nbsp;
  <a href="https://boosty.to/ods_mlcourse">
         <img src="../../_static/img/boosty_logo.png" width=200px >
  </a>
  </p>
  </div>
  <div class="col-md-4" markdown="1">
  <details>
  <summary>Details of the deal</summary>

mlcourse.ai is still in self-paced mode but we offer you Bonus Assignments with solutions for a contribution of $17/month. The idea is that you pay for ~1-5 months while studying the course materials, but a single contribution is still fine and opens your access to the bonus pack.

Note: the first payment is charged at the moment of joining the Tier Patreon, and the next payment is charged on the 1st day of the next month, thus it's better to purchase the pack in the 1st half of the month.

mlcourse.ai is never supposed to go fully monetized (it's created in the wonderful open ODS.ai community and will remain open and free) but it'd help to cover some operational costs, and Yury also put in quite some effort into assembling all the best assignments into one pack. Please note that unlike the rest of the course content, Bonus Assignments are copyrighted. Informally, Yury's fine if you share the pack with 2-3 friends but public sharing of the Bonus Assignments pack is prohibited.
</details>
  </div>
</div><br>

In this assignment, we'll go through the math and code behind decision trees applied to the regression problem, some toy examples will help with that. It is good to understand this because the regression tree is the key component of the gradient boosting algorithm which we cover in the end of the course.

<p float="left">
  <img src="../../_static/img/assignment03_decision_trees_solution_10_0.png" width="350" />
  <img src="../../_static/img/assignment03_decision_trees_solution_17_0.png" width="350" />
</p>

_Left: Building a regression tree, step 1. Right: Building a regression tree; step 3_

Further, we apply classification decision trees to cardiovascular disease data.

<p float="left">
  <img src="../../_static/img/SCORE_CVD_eng.png" width="200" />
  <img src="../../_static/img/assignment03_decision_trees_solution_SCORE.png" width="500" />
</p>

_Left: Risk of fatal cardiovascular disease. Right: A decision tree fit to cardiovascular disease data._

In one more bonus assignment, a more challenging one, you'll be guided through an **implementation of a decision tree from scratch**. You'll be given a template for a general `DecisionTree` class that will work both for classification and regression problems, and then you'll be testing the implementation with a couple of toy- and actual classification and regression tasks.

<div align="center">
<img src="../../_static/img/decision_tree_class_template.png" />
</div>
