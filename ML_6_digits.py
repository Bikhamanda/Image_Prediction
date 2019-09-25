import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression


dataDg = load_digits()
# print(dir(dataDg))
# print(dataDg['data'][0]) # 1x64
# print(dataDg['images'][0]) # 8x8
# print(dataDg['target'][12])

# # # splitting datasets
from sklearn.model_selection import train_test_split
xtr, xts, ytr, yts = train_test_split(
    dataDg['data'],
    dataDg['target'],
    test_size = .1
)
# print(len(xtr))
# print(len(ytr))

# # # Logistic Regression
model = LogisticRegression(
    solver='lbfgs',
    multi_class='auto', # Ditambah "multi_class" karna terlalu banyak feature/ class pada data
    max_iter= 10000
)
model.fit(xtr, ytr)

# # # Visualization + (xtr, ytr)

# fig = plt.figure('LogReg', figsize = (9,3))
# for i in range(10):
#     prediksi = model.predict(xts[0].reshape(1, -1))[0]
#     akurasi = round(model.score(xts, yts)* 100, 2)
#     plt.imshow(xts[i].reshape(8,8), cmap='gray')
#     plt.subplot(2, 5, i+1)
#     plt.title(
#         f'P = {prediksi} | DA = {yts[i]} | A = {akurasi}%'
#     )
# plt.show()

# # # Gambar 8 bit
from PIL import Image 
import PIL.ImageOps
gbr = Image.open('4.jpg').convert('L')
gbr = gbr.resize((8, 8))
gbr = PIL.ImageOps.invert(gbr)
gbrArr = np.array(gbr)
gbrArr2 = gbrArr.reshape(1, 64)
prediksi = model.predict(gbrArr2.reshape(1, -1))
print(prediksi[0])

plt.imshow(gbrArr, cmap='gray')
plt.title('Prediksi : {}'.format(prediksi[0]))
plt.show()