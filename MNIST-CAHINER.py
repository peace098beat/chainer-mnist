
# coding: utf-8

# # Chainer でMINIST
# [Qiita](http://qiita.com/kenmatsu4/items/7b8d24d4c5144a686412)

# まず最初に必要なライブラリ群のインポートです。

# In[120]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_mldata
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions  as F
import sys

plt.style.use('ggplot')


# 次に各種パラメーターの定義・設定を行います。

# In[121]:

# 確率的勾配降下法で学習させる際の１回分のバッチサイズ
batchsize = 100

# 学習の繰り返し回数
n_epoch   = 20

# 中間層の数
n_units   = 1000


# Scikit LearnをつかってMNISTの手書き数字データをダウンロードします。

# In[122]:

# MNISTの手書き数字データのダウンロード
# #HOME/scikit_learn_data/mldata/mnist-original.mat にキャッシュされる
print 'fetch MNIST dataset'
mnist = fetch_mldata('MNIST original', )
print 'end > fetch MNIST dataset'
# mnist.data : 70,000件の784次元ベクトルデータ
mnist.data   = mnist.data.astype(np.float32)
mnist.data  /= 255     # 0-1のデータに変換
print("mnist.data.shape %r"%(mnist.data.shape,))

# mnist.target : 正解データ（教師データ）
mnist.target = mnist.target.astype(np.int32)
print("mnist.target.shape %r"%(mnist.target.shape,))
print(mnist.target[0:70000:1000])


# 3つくらい取り出して描画してみます。

# In[123]:

def draw_digit(data):
    size = 28
    plt.figure(figsize=(2.5,3))
    X,Y = np.meshgrid(range(size), range(size))
    Z = data.reshape(size, size) # convert from vector to 28x28 matrix
    Z = Z[::-1,:] #frip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(X, Y, Z)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
    
    plt.show()
    
import random
I =random.randint(0,70000)
draw_digit(mnist.data[I])


# データセットを学習用データ(train), 検証用データ(test)に分割します。

# In[124]:

# 学習用データをN個, 検証用データを残りの個数と設定
N = 60000 # 学習用データ(traindata)の個数
x_train, x_test = np.split(mnist.data, [N])
y_train, y_test = np.split(mnist.target, [N])
N_test = y_test.size


# In[125]:

print("Training Data X: %r, Test Data X:%r" % (x_train.shape, x_test.shape))
print("Training Data Y: %r, Test Data Y:%r" % (y_train.shape, y_test.shape))
print("Test Data Size: %r" %(N_test,))


# ## モデルの定義
# ここからchainerのクラスや関数を使います

# In[126]:

# Prepare multi-layer perceptron model
# 入力 784次元, 出力10次元
# n_units:中間層のノード数
model = FunctionSet(l1=F.Linear(784, n_units),
                    l2=F.Linear(n_units, n_units),
                    l3=F.Linear(n_units, 10))


# In[127]:

# Neural net architecture
def forward(x_data, y_data, train=True):
    x, t = Variable(x_data), Variable(y_data)
    h1 = F.dropout(F.relu(model.l1(x)), train=train)
    h2 = F.dropout(F.relu(model.l2(h1)), train=train)
    y = model.l3(h2)
    # 多クラス分類なので誤差関数としてソフトマックス関数の，
    # 交差エントロピー関数を用いて，誤差を導出
    err = F.softmax_cross_entropy(y, t), F.accuracy(y, t)
    return err


# 説明
# 活性化関数にはシグモイド関数ではなく，F.relu()関数が使われている．
# F.relu(), 正規化(整流化)線形関数(Rectified Linear Unit function)
# 
# 
# $ f(x) = max(0,x)$

# In[128]:

def plot_relu_function():
    """Plot relu function (Rectified Linear Unit function)
    $ f(x) = max(0,x)$
    """
    x_data = np.linspace(-10, 10, 100, dtype=np.float32)
    x = Variable(x_data)
    y = F.relu(x)
    plt.figure(figsize=(7,5))
    plt.ylim(-2,10)
    plt.plot(x.data, x.data, c="k", label="Linear")
    plt.plot(x.data, y.data, c="r", label="Relu")
    plt.legend()
    plt.show()
plot_relu_function()


# ついでにシグモイド関数もプロットしておく

# In[129]:

def plot_sigmoid_function():
    """Plot Sigmoid function (Rectified Linear Unit function)
    $ f(x) = $
    """
    x_data = np.linspace(-10, 10, 100, dtype=np.float32)
    x = Variable(x_data)
    y = F.sigmoid(x)
    
    plt.figure(figsize=(7,5))
    plt.ylim(-1,2)
    plt.plot(x.data, F.relu(x).data, c="k", label="relu")
    plt.plot(x.data, y.data, c="r", label="Sigmoid")
    plt.legend()
    plt.show()
    
plot_sigmoid_function()


# 次に、このrelu()関数の出力を入力としてF.dropout()関数が使われています。
# 
# ``` F.dropout(F.relu(model.l1(x)),  train=train) ```  
# このドロップアウト関数F.dropout()はDropout: A Simple Way to Prevent Neural Networks from Overfittingという論文で提唱されている手法で、ランダムに中間層をドロップ（ないものとする）し、そうすると過学習を防ぐことができるそうです。

# In[130]:

# dropout(x, ratio=0.5, train=True) テスト
# x: 入力値
# ratio: 0を出力する確率
# train: Falseの場合はxをそのまま返却
# return: ratioの確率で0を, 1-ratioの確率で x*(1/(1-raio))の値を返す
def dropout_test(n=50):
    v_sum = 0
    for i in range(n):
        x_data = np.array([1,2,3,4,5,6], dtype=np.float32)
        x = Variable(x_data)
        drout = F.dropout(x, ratio=0.6, train=True)

        for j in range(6):
            # sys.stdout.write( str(drout.data[j]) + ', ' )
            pass
        # print("")
        v_sum += drout.data
    sys.stdout.write( str((v_sum/float(n))))
    print("v_sumの値がx_dataと大体一致する")
dropout_test(10)
dropout_test(20)
dropout_test(30)
dropout_test(40)


# ソフトマックス関数
# $$ y_k = z_k = f_k(u) = \frac{exp(u_k)}{\sum_{j}^{K} exp(u_j)} $$

# また、F.accuracy(y, t)は出力と、教師データを照合して正答率を返しています。

# 

# ### 3.3 Optimizerの設定  
# さて、モデルが決まったので訓練に移ります。
# ここでは最適化手法としてAdamが使われています。

# In[131]:

# setup optimizer
optimizer = optimizers.Adam()
optimizer.setup(model.collect_parameters())


# ## 訓練の実施と結果

# In[132]:

train_loss = []
train_acc  = []
test_loss = []
test_acc  = []

l1_W = []
l2_W = []
l3_W = []

sys.stdout.write("Start Traning... \n")
sys.stdout.flush()


# Learning loop
for epoch in xrange(1, n_epoch+1):
    sys.stdout.write('epoch : %r \n'% epoch)
    sys.stdout.flush()
    
    # training
    # N個の順番をランダムに並び替える
    perm = np.random.permutation(N)
    sum_accuracy = 0
    sum_loss = 0
    
    # 0〜Nまでのデータをバッチサイズごとに使って学習
    for i in xrange(0, N, batchsize):
        x_batch = x_train[perm[i:i+batchsize]]
        y_batch = y_train[perm[i:i+batchsize]]

        # 勾配を初期化
        optimizer.zero_grads()
        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch)
        # 誤差逆伝播で勾配を計算
        loss.backward()
        optimizer.update()

        train_loss.append(loss.data)
        train_acc.append(acc.data)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # 訓練データの誤差と、正解精度を表示
    print 'train mean loss={}, accuracy={}'.format(sum_loss / N, sum_accuracy / N)

    # evaluation
    # テストデータで誤差と、正解精度を算出し汎化性能を確認
    sum_accuracy = 0
    sum_loss     = 0
    for i in xrange(0, N_test, batchsize):
        x_batch = x_test[i:i+batchsize]
        y_batch = y_test[i:i+batchsize]

        # 順伝播させて誤差と精度を算出
        loss, acc = forward(x_batch, y_batch, train=False)

        test_loss.append(loss.data)
        test_acc.append(acc.data)
        sum_loss     += float(cuda.to_cpu(loss.data)) * batchsize
        sum_accuracy += float(cuda.to_cpu(acc.data)) * batchsize

    # テストデータでの誤差と、正解精度を表示
    print 'test  mean loss={}, accuracy={}'.format(sum_loss / N_test, sum_accuracy / N_test)

    # 学習したパラメーターを保存
    l1_W.append(model.l1.W)
    l2_W.append(model.l2.W)
    l3_W.append(model.l3.W)


print("Finish! Traning")


# In[133]:

# 精度のグラフ描画
plt.figure(figsize=(8,3))
plt.plot(range(len(test_acc)), test_acc, c='b')
plt.legend(["test_acc"],loc=4)
plt.title("Accuracy of digit recognition.")
plt.ylim(0.7,1)
plt.plot()
# 精度のグラフ描画
plt.figure(figsize=(8,3))
plt.plot(range(len(train_acc)), train_acc, c='r')
plt.legend(["train_acc"],loc=4)
plt.title("Accuracy of digit recognition.")
plt.ylim(0.7,1)
plt.plot()


# In[134]:

# 誤差のグラフ描画
plt.figure(figsize=(8,3))
plt.plot(range(len(train_loss)), train_loss, c='b')
plt.legend(["train_loss"],loc=4)
plt.title("Loss of digit recognition.")
plt.ylim(0.0, 1)
plt.plot()
# 誤差のグラフ描画
plt.figure(figsize=(8,3))
plt.plot(range(len(test_loss)), test_loss, c='r')
plt.legend(["test_loss"],loc=4)
plt.title("Loss of digit recognition.")
plt.ylim(0.0,1)
plt.plot()


# ### 答え合わせ
# 識別した100個の数字を表示してみます。ランダムに100個抽出したのですが、ほとんど正解です。何回か100個表示を行ってやっと間違っているところを１つ表示できたので、その例を下記に貼っています。なんだか人間の方が試されている気分です（笑）

# In[135]:

plt.style.use('fivethirtyeight')
def draw_digit3(data, n , ans, recog):
    size = 28
    plt.subplot(10,10,n)
    Z = data.reshape(size,size)
    Z = Z[::-1, :]
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("ans=%d, recog=%d"%(ans, recog), size=8)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")
plt.figure(figsize=(15,15))


cnt = 0
# for idx in np.random.permutation(N)[:1000]:
for idx in range(N):
    if mod(idx , 1000):
        pass
        
    xxx = x_train[idx].astype(np.float32)
    h1 = F.dropout(F.relu(model.l1(Variable(xxx.reshape(1,784)))), train=False)
    h2 = F.dropout(F.relu(model.l2(h1)), train=False)
    y = model.l3(h2)
    
    # 間違えだけ表示
    if y_train[idx] != np.argmax(y.data):
        cnt += 1
        draw_digit3(x_train[idx], cnt, y_train[idx], np.argmax(y.data))
plt.show()
print("Fin")


# ## 第一層パラメータWの可視化

# In[140]:

def draw_digit2(data, n, i):
    size = 28
    plt.subplot(10, 10, n)
    Z = data.data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,27)
    plt.ylim(0,27)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

plt.figure(figsize=(10,10))
cnt = 1
for i in np.random.permutation(1000)[:100]:
    draw_digit2(l1_W[len(l1_W)-1][i], cnt, i)
    cnt += 1

plt.show()
print("Fin")


# In[142]:

# レイヤー3
def draw_digit2(data, n, i):
    data = data.data
    size = 32
    plt.subplot(4, 4, n)
    data = np.r_[data,np.zeros(24)]
    Z = data.reshape(size,size)   # convert from vector to 28x28 matrix
    Z = Z[::-1,:]             # flip vertical
    plt.xlim(0,size-1)
    plt.ylim(0,size-1)
    plt.pcolor(Z)
    plt.title("%d"%i, size=9)
    plt.gray()
    plt.tick_params(labelbottom="off")
    plt.tick_params(labelleft="off")

plt.figure(figsize=(10,10))
cnt = 1
for i in range(10):
    draw_digit2(l3_W[len(l3_W)-1][i], cnt, i)
    cnt += 1

plt.show()


# In[143]:

# 活性化関数テスト
x_data = np.linspace(-10, 10, 100, dtype=np.float32)
x = Variable(x_data)

y = F.relu(x)
plt.figure(figsize=(8,15))
plt.subplot(311)
plt.title("ReLu function.")
plt.ylim(-2,10)
plt.xlim(-6,6)
plt.plot(x.data, y.data)

y = F.tanh(x)
plt.subplot(312)
plt.title("tanh function.")
plt.ylim(-1.5,1.5)
plt.xlim(-6,6)
plt.plot(x.data, y.data)

y = F.sigmoid(x)
plt.subplot(313)
plt.title("sigmoid function.")
plt.ylim(-.2,1.2)
plt.xlim(-6,6)
plt.plot(x.data, y.data)
plt.show()

