# import libs
import pandas as pd
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from flask import Flask

app = Flask(__name__)


def classificator(data_frame):  #Определяем функицю
    for feature in list(data_frame.columns):      # Задаем итератор в рамках названий колонок
        if data_frame[feature].dtype == 'O':      # Условие для признаков соответсвующих типу даных "Objekt"
            data_frame[feature].replace(['Yes', 'No'], [1, 0], inplace=True)   # Замена значений 'Yes' и 'No' на 0 и 1
            for iteration, value in enumerate(list(data_frame[feature].unique())):  # Итератор в рамках уникальных значений признака
                if type(value) == str:         # Условие для замены нецифровых значений
                    if data_frame[feature].nunique() > 2:  # Условие компенсации порядкового номера для
                        iteration += 1                     # тех признаков, в которых не было значений 'Yes' и 'No'
                    data_frame[feature].replace(value, iteration, inplace=True)  # Замена всех нецифровых значений на соответсвущй порядковый номер


@app.route('/predict')
def predict():
    # read data
    data = pd.read_csv('~/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # prepare data
    del data['customerID']

    new_type_list = []
    for i in data['TotalCharges']:
        try:
            i = float(i)
        except:
            i = 0
        new_type_list.append(i)
    data['TotalCharges'] = new_type_list

    classificator(data)

    data["Churn"] = data["Churn"].astype(int)

    # split data to test and train
    Y = data['Churn']
    X = data.drop(labels=["Churn"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=.3, stratify=Y, random_state=17)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)
    # create, train & predict model
    lr = LogisticRegression()
    model = lr.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    # examine the impact of every property from dataset
    weights = pd.Series(model.coef_[0], index=X.columns.values)
    weights = weights.sort_values(ascending=False)
    # print results
    accuracy = metrics.accuracy_score(y_test, prediction_test)
    print(accuracy)
    print(weights)
    result = '<h1>' + str(accuracy) + '\n'
    for weight in weights:
        result += str(weight) + "\n"
    result += '</h1>'
    return result

if __name__ == '__main__':
    app.run()
