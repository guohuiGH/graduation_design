# graduation_design

#guohui

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
#在human目录下已经有了训练数据集(train)和测试集（test）,gbdt_lstm_human.py为gbdt+lstm模型,lstm.py,gbdt_human.py为单独的gbdt和human模型，这三个模型都会通过script/目录下的read_data.py读取数据
#在wafer目录下数据集和测试集在同级目录UCR_TS_Archive_2015，里面， wafer.py为gbdt+lstm模型，lstm_wafer.py, gbdt_wafer.py为单独的模型，这三个模型都会通过script/目录下的read_data.py读取数据
#向read_data传入的参数主要是文件目录，输入的时间长度，分类的类别。
#gbdt+lstm新模型会关联到script/gbdt_net.py(用于生成森林层),script/validation.py（评价结果）


