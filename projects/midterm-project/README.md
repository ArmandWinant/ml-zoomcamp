# Machine Learning Bootcamp 2024
## Midterm Project
### Introduction
In this project, I trained a binary classification model to predict the likelihood that a Himalaya climber's expedition will be successful. The dataset used for model training can be found online ([www.himalayandatabase.com](https://www.himalayandatabase.com/)), and covers all expeditions from 1905 through Autumn-Winter 2023 to the most significant mountaineering peaks in Nepal.

### Model Training
#### Data Preparation
##### Raw Data Import

The data is imported from three separate files

1. _member.DBF:_ details about the climber at the time of the expedition (age, gender, status,...)
2. _exped.DBF:_ information about the entire expedition team (number of members, route, equipment,...)
3. _peaks.DBF:_ facts about the mountain (height, location,...)

##### Variable Selection

The first data cleaning step consisted of selecting only the columns that could be used to predict the success of an inidividual climber's upcoming expedition.

Many of the variables in the raw dataset related to post-expedition information, which could not be included as part of any model's input features.

I performed the variable selection manually by referring to the provided data dictionaries.

##### Data Filtering

In order to maximize the odds of achieving a robust model with significant predictive power, the project score was narrowed as below:

1. only members with intention to submit werre considered, which excluded sherpas, photographers, cooks, etc.
2. expeditions that involved non-climbing components (ski, parapente, traverse) were excluded
3. data prior to the year 2000 was disregarded on the count of climbing practices having evolved


##### Feature Engineering

The following variables were added through basic computations:

1. __member_experience:__ the number of expeditions the member had participated in prior the current expedition
2. __agency experience:__ the number of expeditions completed by the parterning agency prior to the current expedition.

#### Model Training
##### Models

I trained and compared three models:

1. Logistic Regression
2. Decision Tree
3. Random Forest

Throughout training, I used the accuracy metric as the criterion for evaluating and comparing models.

##### Feature Selection

I performed rudimentary feature selection by training separate models on different subsets of the original features.

Across models, the accuracy peformance was not significantly affected by the removal of individual features from the data.

##### Parameter Tuning

Using basic loops, I optimized the models with regards to specific model parameters. The accuracy of each model was improved by several percentage as a result of the optimization process.

##### Outcome

Out of all three models, the decision tree classifier had the highest accuracy score on the test data and was used as the final model.


### How to run
1. Install pipenv from the terminal: `pip install pipenv`
2. Run all the cells in `notebook.ipynb` to train models
3. Train the final model from the terminal: `python train.py`
4. Build the docker image from the terminal: `docker build -t zoomcamp-midterm .`
5. Start the docker container from the terminal: `docker run -it --rm -p 9696:9696 zoomcamp-midterm:latest`
6. In a different terminal windown, query the flask webservice: `python test-predict.py`