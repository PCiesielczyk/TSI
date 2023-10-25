import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from data_storage.file_loader import X_train, X_val, y_train, y_val

print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

plt.figure(0, figsize=(10, 5))
unique_train, counts_train = np.unique(y_train, return_counts=True)
plt.bar(unique_train, counts_train)

plt.title('Training Set Class Distribution', fontsize=22)
plt.xlabel('Class Number', fontsize=18)
plt.ylabel('Number of Occurrences', fontsize=20)
plt.tick_params(labelsize=16)
plt.grid(linestyle=':')
plt.show()
