(bonus04)=

# Bonus Assignment 4. Beating baselines in a competition

```{figure} /_static/img/topic4-teaser.png
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

In this assignment, you’ll be guided through working with sparse data, feature engineering, model validation, and the process of competing on Kaggle. The task will be to beat baselines in the “Alice” [Kaggle competition](https://www.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking2). That’s a very useful assignment for anyone starting to practice with Machine Learning, regardless of the desire to compete on Kaggle.

The competition is about identifying a user ("Alice") on the Internet through tracking her web sessions, it's based on the actual data from one French University. The competition turned out to be very successful, in a sense that the task can be solved well with fairly simple models (literally, logistic regression), and extensive feature engineering.


```{figure} /_static/img/topic6-teaser.png
:width: 600px
```

For example, the figure above depicts the distribution of session start hours for Alice and others. You might see that distributions are quite different. Hence, such a feature can be added to the model and improve its quality. Such an activity -- feature engineering -- is a very creative process (we touch it later in the course as well). And it turns out, this competition is very rewarding for thoughtful feature engineering. And well, coming up with good features via visual analysis, adding those to the model, and climbing up the leaderboard -- that's an unforgettable adventure!

In this task, we arm you with a fairly well-performing baseline, and then you are invited to come up with new features and beat baselines.
