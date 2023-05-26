#importing all the required libraries
import pandas as pd
import numpy as np
import sklearn as sk
import pickle
from sklearn.feature_extraction.text import CountVectorizer

def trainModel():
    df = pd.read_csv('deceptive-opinion.csv')

    #Extracting only the requireed features
    df1 = df[['deceptive', 'text']]

    #filling the categorical variable deceptive with 0 for fake review and 1 for real review
    df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
    df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1

    #Taking the input and output features seperately
    X = df1['text']
    Y = np.asarray(df1['deceptive'], dtype = int)

    #importing MultinomialNB
    from sklearn.naive_bayes import MultinomialNB, GaussianNB

    #splitting the data into training and testing set  with test size is 30%
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3,random_state=109) # 70% training and 30% test

    X_test.to_csv('test.csv')

    nb = MultinomialNB()

    #Converting the review (text feature) to numerical features
    cv = CountVectorizer()
    x = cv.fit_transform(X_train)
    y = cv.transform(X_test)

    # Fitting the model
    nb.fit(x, y_train)
    pickle.dump(nb,open('model.pkl','wb'))
    model=pickle.load(open('model.pkl','rb'))

    print(nb.predict(x))

    # Training Accuracy
    print(nb.score(x, y_train))

    # Testing Accuracy
    print(nb.score(y, y_test))

    return(nb.score(x, y_train),nb.score(y, y_test))