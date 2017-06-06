## Открытый курс OpenDataScience по машинному обучению
![ODS stickers](https://github.com/Yorko/mlcourse_open/blob/master/img/ods_stickers.jpg)

В курсе даются теоретические основы "классического" машинного обучения, а также с помощью обилия домашних заданий и 2 соревнований Kaggle Inclass можно приобрести навыки практического анализа данных и построения прогнозных моделей. 
Требуются начальные навыки программирования на Python и знание математики (математический анализ, линейная алгебра, теория вероятностей, математическая статистика) на уровне 2 курса технического ВУЗа. [Wiki-страница](https://github.com/Yorko/mlcourse_open/wiki/Prerequisites:-Python,-математика,-DevOps) с информацией, как лучше подготовиться к прохождению курса, если навыков программирования или знаний математики не хватает.

Ниже перечислены основные темы (со ссылками на статьи на Хабре), домашние задания и соревнования Kaggle в рамках курса. На [Wiki](https://github.com/Yorko/mlcourse_open/wiki) можно найти информацию об авторах статей, необходимом для прохождения курса ПО, а также можно увидеть список тьюториалов, написанных участниками курса, по темам, связанным с DS & ML.

1 итерация курса прошла с 28 февраля по 10 июня 2017 года – с домашними заданими, соревнованиями, тьюториалами, конкурсами по визуализации, общим рейтингом (список топ-100 участников скоро будет опубликован на [Wiki](https://github.com/Yorko/mlcourse_open/wiki)). Было весело :grinning:. Повторение курса не планируется, но можно самоорганизоваться, если найдутся энтузиасты. 

## Основные темы
1. [Первичный анализ данных с Pandas](https://habrahabr.ru/company/ods/blog/322626/)
2. [Визуальный анализ данных с Python](https://habrahabr.ru/company/ods/blog/323210/)
3. [Классификация, деревья решений и метод ближайших соседей](https://habrahabr.ru/company/ods/blog/322534/)
4. [Линейные модели классификации и регрессии](https://habrahabr.ru/company/ods/blog/323890/)
5. [Композиции: бэггинг, случайный лес](https://habrahabr.ru/company/ods/blog/324402/)
6. [Построение и отбор признаков](https://habrahabr.ru/company/ods/blog/325422/)
7. [Обучение без учителя: PCA и кластеризация](https://habrahabr.ru/company/ods/blog/325654/)
8. [Обучаемся на гигабайтах с Vowpal Wabbit](https://habrahabr.ru/company/ods/blog/326418/)
9. [Анализ временных рядов с помощью Python](https://habrahabr.ru/company/ods/blog/327242/)
10. [Градиентный бустинг. Часть 1](https://habrahabr.ru/company/ods/blog/327250/) 
11. Градиентный бустинг. Часть 2. ~ 20.06.2017

## Домашние задания
1. Анализ данных по доходу населения UCI Adult. [Тетрадка](https://goo.gl/RjJlYR), [веб-форма](https://goo.gl/forms/63kYBviuDJuFz24E2) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic01_pandas_data_analysis/%5Bsolution%5D_hw1_adult_pandas.ipynb)
2. Визуальный анализ данных о публикациях на Хабрахабре. [Тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic02_visual_analysis/hw2_habr_visual_analysis.ipynb), [веб-форма](https://goo.gl/forms/p8x0SGmn91VCNB6o2) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic02_visual_analysis/%5Bsolution%5D_hw2_habr_visual_analysis.ipynb)
3. Деревья решений в игрушечной задаче и на данных Adult репозитория UCI. [Тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic03_decision_trees_knn/hw3_decision_trees.ipynb), [веб-форма](https://goo.gl/forms/eTz36gkL88QQSzct2) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic03_decision_trees_knn/%5Bsolution%5D_hw3_decision_trees.ipynb)
4. Линейные модели классификации и регрессии в соревнованиях Kaggle Inclass. [Часть 1: идентификация взломщика](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic04_linear_models/hw4_part1_websites_logistic_regression.ipynb),  [Часть 2: прогноз популярности статьи на Хабрахабре](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic04_linear_models/hw4_part2_habr_popularity_ridge.ipynb), [веб-форма](https://goo.gl/forms/6ii1zGEnfJvXhy6E2) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic04_linear_models/%5Bsolution%5D_hw4_part1_websites_logistic_regression.ipynb) 1 части, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic04_linear_models/%5Bsolution%5D_hw4_part2_habr_popularity_ridge.ipynb) 2 части
5. Логистическая регрессия и случайный лес в задаче кредитного скоринга. [Тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic05_bagging_rf/hw5_logit_rf_credit_scoring.ipynb), [веб-форма](https://docs.google.com/forms/d/e/1FAIpQLSdUPWLr5N3YQ1aUpJQGcuJ5UrqUe19rIncpgRLxxlS_XMaUxA/viewform?c=0&w=1) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic05_bagging_rf/%5Bsolution%5D_hw5_logit_rf_credit_scoring.ipynb)
6. Работа с признаками. [Тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic06_features/hw6_features.ipynb), [веб-форма](https://goo.gl/forms/1aSusaXaYm7T422o2) для ответов, [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic06_features/%5Bsolution%5D_hw6_features.ipynb)
7. Метод главных компонент, t-SNE и кластеризация. [Тетрадка](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/mlcourse_open/master/jupyter_notebooks/topic07_unsupervised/hw7_pca_tsne_clustering.ipynb), [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic07_unsupervised/%5Bsolution%5D_hw7_pca_tsne_clustering.ipynb), [веб-форма](https://docs.google.com/forms/d/e/1FAIpQLSdjgje8qvptEW1EKY-QBbHXmXGIs6QYK2PqhchzF4Kpg3v8OQ/viewform) для ответов
8. Часть 1: Реализация алгоритмов онлайн-обучения, [тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic08_sgd_hashing_vowpal_wabbit/hw8_part1_implement_sgd.ipynb), [решение](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/mlcourse_open/master/jupyter_notebooks/topic08_sgd_hashing_vowpal_wabbit/%5Bsolution%5D_hw8_part1_implement_sgd.ipynb). Часть 2: Vowpal Wabbit в задаче классификации тегов вопросов на Stackoverflow, [тетрадка](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/mlcourse_open/master/jupyter_notebooks/topic08_sgd_hashing_vowpal_wabbit/hw8_part2_vw_stackoverflow_tags_10mln.ipynb), [решение](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic08_sgd_hashing_vowpal_wabbit/%5Bsolution%5D_hw8_part2_vw_stackoverflow_tags_10mln.ipynb). [веб-форма](https://goo.gl/forms/8855OkG6em04f8qq1) для ответов
9. Предсказание числа просмотров вики-страницы. [Тетрадка](http://nbviewer.jupyter.org/github/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic09_time_series/hw9_time_series.ipynb), [решение](http://nbviewer.ipython.org/urls/raw.github.com/Yorko/mlcourse_open/master/jupyter_notebooks/topic09_time_series/%5Bsolution%5D_hw9_time_series.ipynb), [веб-форма](https://goo.gl/forms/ywD9QxXsQ3sZEXtu1) для ответов
10. Реализация градиентного бустинга. [Тетрадка](https://goo.gl/THbz1s), [решение](https://goo.gl/tlv53N), [веб-форма](https://goo.gl/forms/mMUhGSDiOHJI9NHN2) для ответов

## Соревнования Kaggle Inclass
1. [Прогноз популярности статьи на Хабре](https://inclass.kaggle.com/c/howpop-habrahabr-favs)
2. [Идентификация взломщика по последовательности переходов по сайтам](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking)






