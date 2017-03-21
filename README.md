# Открытый курс OpenDataScience по машинному обучению
![ODS stickers](https://github.com/Yorko/mlcourse_open/blob/master/img/ods_stickers.jpg)

## Основные темы

1. [Первичный анализ данных с Pandas](https://habrahabr.ru/company/ods/blog/322626/)
2. [Визуальный анализ данных с Python](https://habrahabr.ru/company/ods/blog/323210/)
3. [Классификация, деревья решений и метод ближайших соседей](https://habrahabr.ru/company/ods/blog/322534/)
4. [Линейные модели классификации и регрессии](https://habrahabr.ru/company/ods/blog/323890/)
5. Композиции: бэггинг, случайный лес. 
6. Обучение без учителя: PCA, кластеризация, поиск аномалий
7. Искусство построения и отбора признаков. Приложения в задачах обработки текста, изображений и гео-данных

## Домашние задания
1. [Анализ данных по доходу населения UCI Adult](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic1_pandas_data_analysis/hw1_adult_pandas.ipynb), [веб-форма](https://goo.gl/forms/63kYBviuDJuFz24E2) для ответов, [решение](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic1_pandas_data_analysis/%5Bsolution%5D_hw1_adult_pandas.ipynb)
2. [Визуальный анализ данных о публикациях на Хабрахабре](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic2_visual_analysis/hw2_habr_visual_analysis.ipynb), [веб-форма](https://goo.gl/forms/p8x0SGmn91VCNB6o2) для ответов, [решение](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic2_visual_analysis/%5Bsolution%5D_hw2_habr_visual_analysis.ipynb)
3. [Деревья решений в игрушечной задаче и на данных Adult репозитория UCI](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic3_decision_trees_knn/hw3_decision_trees.ipynb), [веб-форма](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic3_decision_trees_knn/hw3_decision_trees.ipynb) для ответов
4. Линейные модели классификации и регрессии в соревнованиях Kaggle Inclass. [Часть 1: идентификация взломщика](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic4_linear_models/hw4_part1_websites_logistic_regression.ipynb),  [Часть 2: прогноз популярности статьи на Хабрахабре](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic4_linear_models/hw4_part2_habr_popularity_ridge.ipynb), [веб-форма](https://goo.gl/forms/6ii1zGEnfJvXhy6E2) для ответов

## Авторы статей и лекторы 
*(в скобках – ники в OpenDataScience и на Хабрахабре)*


#### Юрий Кашницкий (@yorko, [yorko](https://habrahabr.ru/users/yorko/))
Программист-исследователь Mail.ru Group, старший преподаватель факультета компьютерных наук ВШЭ, научный сотрудник 
Международной научно-учебной лаборатории интеллектуальных систем и структурного анализа ВШЭ. В прошлом — разработчик Hadoop, бизнес-аналитик и Java-программист РДТЕХ. Домашняя страница https://yorko.github.io/ <br>
Преподаватель в годовой [программе](https://cs.hse.ru/dpo/bigml) дополнительного образования по анализу данных в ВШЭ, автор Capstone проекта [специализации](https://www.coursera.org/specializations/machine-learning-data-analysis) Яндекса и МФТИ "Машинное обучение и анализ данных"  <br>
У Юрия есть [репозиторий](https://github.com/Yorko/python_intro) с Jupyter-тетрадками по языку Python и основным алгоритмам и структурам данных

#### Павел Нестеров (@mephistopheies, [mephistopheies](https://habrahabr.ru/users/mephistopheies/))
Data Scientist в стартапе, который нельзя называть. Раньше - программист-исследователь Mail.Ru Group в департаменте рекламы, позже в департаменте поиска. Преподавал в Техносфере@Mail.Ru на базе МГУ ВМК. Еще раньше - программист-исследователь в сфере компьютерного зрения, до нейросетевой эпохи, в Aspose ltd. Домашняя страница http://pavelnesterov.info/  <br>
Павел пишет содержательные [статьи](https://habrahabr.ru/users/mephistopheies/topics/) на Хабре по нейронным сетям.

#### Екатерина Демидова (@katya, [cotique](https://habrahabr.ru/users/cotique/))
Data Scientist в Segmento, г. Санкт-Петербург. Ментор [специализации](https://www.coursera.org/specializations/machine-learning-data-analysis) Яндекса и МФТИ "Машинное обучение и анализ данных"  <br>
У Кати есть [репозиторий](https://github.com/demidovakatya/vvedenie-mashinnoe-obuchenie) со списком книг/курсов/статей по Data Science

#### Мария Мансурова (@miptgirl, [miptgirl](https://habrahabr.ru/users/miptgirl/))
Аналитик-разработчик в команде Яндекс.Метрики. До этого в Яндексе работала аналитиком ключевых показателей. В прошлом также успела поработать бизнес-аналитиком в компании-интеграторе в сфере телекоммуникаций. <br>

#### Арсений Кравченко (@arsenyinfo)

#### Дмитрий Сергеев (@dmitryserg)

#### Виталий Радченко (@vradchenko)

#### Сергей Королев (@libfun)

# Инструкция по установке Docker-контейнера 
*(необходимое ПО)*

В курсе используется сборка библиотек Anaconda, тетрадки Jupyter, Xgboost, Vowpal Wabbit и некоторые другие библиотеки. Все это можно не устанавливать, а использовать Docker-контейнер (требования: около 4 Гб места на диске, 4 Гб RAM). [Введение](https://habrahabr.ru/post/310460/) в Docker. Рекомендуется тем, кто использует Windows, c \*NIX проще самостоятельно установить необходимое (см. Dockerfile). 

Инструкция:
- скачать данный репозиторий
- на Windows скорее всего придется [включить](http://www.sysprobs.com/disable-enable-virtualization-technology-bios) в BIOS виртуализацию, если раньше не использовали виртуальные машины или Docker
- установить [Docker](https://docs.docker.com/engine/installation/)
- установить [Docker Compose](https://docs.docker.com/compose/install/)
- перейти в командной строке/терминале в скачанный каталог mlcourse_open
- выполнить docker-compose up. Первый раз это займет продолжительное время
- открыть localhost:7777 (в файле *docker-compose.yml* можно поменять порт 7777 на любой другой)
- далее можно выполнить тетрадку [check_docker.ipynb](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/check_docker.ipynb) и убедиться, что нужные библиотеки подключаются

Контейнеры Docker, как правило, занимают много места на диске.
- *docker ps* – посмотреть весь список контейнеров
- *docker stop $(docker ps -a -q)* – остановить все контейнеры
- *docker rm $(docker ps -a -q)* – удалить все контейнеры
- *docker images* - посмотреть весь список образов
- *docker rmi \<image_id\>* – удалить ненужный образ

Доступная и понятная [документация](https://docs.docker.com/engine/getstarted/) Docker с примерами
