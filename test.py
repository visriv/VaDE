from utils import *
X,Y = load_data('bing_query')
batch_size = 100
X = X[:len(X) - len(X) % batch_size]
#print('bing: ', X[0][0], X[0][1], X[0][0] + X[0][1]) 


print(X[0])