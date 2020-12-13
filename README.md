# My Solution to atmaCup #8 (private 29th, public 31st)

Competition Page: [atmaCup #8](https://www.guruguru.science/competitions/13)

## Solution Summary

- Models: Linear Stacking of GBDTs (LGB, CatB, XGB)

- Validation Strategy: StratifiedKFold, using the binned target (`Global_Sales`)

- Features:
~1,500 features in total, mainly:

    - PCA, NMF, LDA for co-occurence matrices for categorical features
    - word2vec for categorical features ([Category2VecWithW2V](https://github.com/Ynakatsuka/kaggle_utils/blob/master/kaggle_utils/features/category_embedding.py#L235))
    - tf-idf to pca for `Name` (using [texthero](https://texthero.org/))
    - aggregation (mean, min, max etc) for numerical features (`Critic_score` etc) by categorical features (`Platform` etc) 
    - further statistics computed from the abovementioned aggregation features (`z-score`, `max to min ratio` etc)
    - nunique by categorical features for the other categorical features

It turned out that EDA (Exploratory Data Analysis) is the key to win this competition, as is always the case with a tabular data science competition!

## Usage

### Folders

Confirm that the following folders are located in the exact manner. Add the competition data into the ```input``` folder.

```
.
├── script
├── config
├── output
└── input

```

### How to Use

I use my local Mac, employing 

- [docker](https://www.docker.com/)
- [mlflow](https://mlflow.org/)
- [hydra](https://hydra.cc/)

to manage my experiments.

Run the followings in the docker container (which is mentioned later) to get a submission csv.

```
python script/feature_engineering.py
python script/fit_predict.py
```

## Environment

### Docker

Install ```docker / docker-compose``` in your machine.

```bash
(base) docker-compose up -d --build  # build a container
(base) docker exec -it katsu1110-atmacup8 bash # get in a container

```

- By typing `jupyter lab` inside the container **jupyter lab** can be initiated in `http://localhost:8888/`.
- By typing `mlflow ui` inside the container **mlflow ui** can be initiated in `http://localhost:5000/`.