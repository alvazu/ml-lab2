import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
 
def normalize_feature(df):
#     """Applies function along input axis(default 0) of DataFrame."""
   return df.apply(lambda column: (column - column.mean()) / column.std())
 
def lr_cost(w, x, y):
   n = x.shape[0]#n为样本数
 
   inner = x @ w - y
 
   square_sum = inner.T @ inner
   w_ = w[1:n]
   w_ = w_.T @ w_ - 1
   w_ = w_ * w_
   cost = square_sum / (2 * n) + w_
 
   return cost
def gradient(w, x, y):
   n = x.shape[0]
   temp_w = w[1:n]
   temp = temp_w.T @ temp_w - 1
   inner = x.T @ (x @ w - y)
   temp_w = 4*temp * temp_w
   w = np.concatenate([np.array([0]),temp_w])
   inner = inner /n + w
   return inner
 
def batch_gradient_decent(w, x, y, epoch, alpha=0.01):
 
   cost_data = [lr_cost(w, x, y)]
   for _ in range(epoch):
       w = w - alpha * gradient(w,x,y)
       cost_data.append(lr_cost(w, x, y))
   return w, cost_data
 
 
data_path = "ex1data2.txt"
 
data = pd.read_csv(data_path, names=['1', '2','3'])
data = normalize_feature(data)
ones = pd.DataFrame({'0':np.ones(len(data))})
 
data = pd.concat([ones,data],1)
 
x = data.iloc[:,:-1].values
y = np.array(data.iloc[:,-1])
w = np.zeros(x.shape[1])
 
final_theta, cost_data = batch_gradient_decent(w, x, y, 500)
print(final_theta[0:10])
