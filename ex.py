import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

filepath = "/Users/navi/Desktop/Kaggle/imdb/"
dataFileName = "imdb.csv"

df = pd.read_csv(filepath+dataFileName)

y = df['SciFi']

X_train, X_test, y_train, y_test = train_test_split(
                                        df['plot'], y
                                        test_size = 0.33,
                                        random_state = 53
                                        )
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train.values)
count_test = count_vectorizer.transform(X_test.values)
