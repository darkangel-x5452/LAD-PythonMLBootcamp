from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential


def sample_1():
    df = pd.read_csv(
        '../../Refactored_Py_DS_ML_Bootcamp-master/22-Deep Learning/TensorFlow_FILES/DATA/cancer_classification.csv')
    X = df.drop('benign_0__mal_1', axis=1).values
    y = df['benign_0__mal_1'].values
    print(X)
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.25, random_state=101)
    scaler = MinMaxScaler()
    scaler.fit(xtrain)
    xtrain = scaler.transform(xtrain)
    xtest = scaler.transform(xtest)

    early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
    ts = datetime.now().strftime('%Y-%m-%d--%H%M')
    log_dir = "deep_learning_tensorboard_logs/fit"
    log_dir = log_dir + "/" + ts

    board = TensorBoard(log_dir=log_dir, histogram_freq=1,
                        write_graph=True,
                        write_images=True,
                        update_freq='epoch',
                        profile_batch=2,
                        embeddings_freq=1)

    model = Sequential()
    model.add(Dense(units=30, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=15, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam')

    model.fit(x=xtrain, y=ytrain,
              epochs=600,
              validation_data=(xtest, ytest),
              verbose=1,
              callbacks=[early_stop, board])

    plt.show()
    print('bye')


if __name__ == '__main__':
    sample_1()
