# graduation_design
#author:guohui
#mail:guohui1029@foxmail.com

#数据集准备
#下载数据集wafer，star数据集在ucr http://www.cs.ucr.edu/~eamonn/time_series_data/
#下载uci数据集,human activity https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
#服务器上也有对应的数据，ucr地址home/guohui/graduation_design/UCR_TS_Archive_2015,uci的humanActivity数据集 huamn目录下

#安装环境准备
#目前我使用的是anaconda+tensorflow0.12+pandas+sklearn0.18+xgboost
#tensorflow,skelarn 的版本建议向上按照，tensorflow目前有1.2版本（2017.5.30）
#目前服务器GPU的环境为cuda7.5，tensorflow1.0以后要求cuda8.0，要使用GPU建议通过anaconda安装0.11版本的tensorflow

#以下安装以linux为例
#anaconda download + install
wget https://repo.continuum.io/archive/Anaconda2-4.3.1-Linux-x86_64.sh 
bash Anaconda2-4.3.1-Linux-x86_64.sh 

#tensorflow1.1+pandas+skelarn down+install
#创建tensorflow环境名
conda create -n tensorflow
#激活环境
source activate tensorflow
#安装python2.7+tesnorflow1.1，要改用其他版本的，可以改掉后面的url链接，具体其他链接见https://www.tensorflow.org/install/install_linux#the_url_of_the_tensorflow_python_package
pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.1.0-cp27-none-linux_x86_64.whl
#检查是否安装成功：
>>python
import tensorflow
#在tensoflow环境继续安装pandas，sklearn,
conda install pandas
conda install scikit-learn
#或者用pip安装
pip install pandas
pip install -U scikit-learn
#关闭环境
source deactivate tensorflow

#使用说明，以huamn，wafer为例
#先激活环境 
source activate tensorflow
#在human目录下已经有了训练数据集(train)和测试集（test）,
  #human/gbdt_lstm_human.py为gbdt+lstm模型;
  #huam/lstm.py,human/gbdt_human.py为单独的gbdt和human模型;
  #三个模型都会通过script/目录下的read_data.py读取数据.
#在wafer目录下数据集和测试集在同级目录UCR_TS_Archive_2015，里面， 
  #wafer/wafer.py为gbdt+lstm模型;
  #wafer/lstm_wafer.py, wafer/gbdt_wafer.py为单独的模型；
  #三个模型也都会通过script/目录下的read_data.py读取数据
#向read_data传入的参数主要是文件目录，输入的时间长度，分类的类别。
#gbdt+lstm新模型会关联到tensorflow/gbdt_net.py(用于生成森林层),script/validation.py（评价结果）

#相关文件使用说明
#tensorflow/gbdt_lstm_libsvm.py :手写数据集训练模型,在确定数据
#tensorflow/gpu_gbdt_lstm_libsvm.py：gpu版的手写数据集，需要改变训练环境(使用GPU训练时间能大幅度下降)
#tensorflow/lstm_gbdt_feature.py: 用gbdt先生成0，1特征成，再放入到lstm训练，不是端到端的模型（实际效果比gbdt+lstm端到端的模型差）
#tensorlfow/gbdt_net.py: 用于生成训练的森林层网络
#tensorflow/lstm_libsvm_example.py: 单独lstm网络的手写数据集的训练模型
#script/read_data.py: 用于读取文件，并进行相关的预处理
#script/validation.py: 用于评价模型预测的结果

