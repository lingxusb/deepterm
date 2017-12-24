from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import matplotlib.pylab as plt

def seq_matrix(seq_list):
	tensor = np.zeros((len(seq_list),100,4))
	for i in range(len(seq_list)):
		seq = seq_list[i]
		j = 0
		for s in seq:
			if s == 'A':
				tensor[i][j] = [1,0,0,0]
			if s == 'T':
				tensor[i][j] = [0,1,0,0]
			if s == 'C':
				tensor[i][j] = [0,0,1,0]
			if s == 'G':
				tensor[i][j] = [0,0,0,1]
			j = j + 1
	return tensor

seq = open('train_seq.txt').readlines()
flux = np.loadtxt('train_v.txt')
print(len(seq))
print(seq[0])
print(len(flux))

print('Building model...')
model = Sequential()
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=4,
                        input_length=100,
                        #border_mode='valid',
                        #W_constraint = maxnorm(3),
                        activation='relu'))
                        #subsample_length=1))
model.add(MaxPooling1D(pool_length=3))
#model.add(Convolution1D(nb_filter=128,
#						filter_length=3,
#						input_dim=4,
#                        input_length=100,
#                        activation='relu'
#						))
model.add(Dropout(p=0.21370950078747658))
model.add(LSTM(output_dim=128,
               return_sequences=True))
#model.add(Dropout(p=0.7238091317104384))
#model.add(Dense(128,activation = 'relu'))
model.add(Dropout(p=0.2))
model.add(Flatten())
model.add(Dense(1,activation = 'relu'))
#model.add(Dropout(p=0.5))
#odel.add(Activation('sigmoid'))

print('Compiling model...')
model.compile(loss='mean_squared_error',
              optimizer='sgd',
metrics=['accuracy'])


X_test = seq_matrix(seq)
print(X_test[0,:])
model.fit(X_test,np.log(flux+1), verbose = 1, validation_split=0.1,epochs=10)
print('train completed')
y_test = model.predict(X_test)
plt.plot(np.log(flux+1), y_test,'.')
plt.show()