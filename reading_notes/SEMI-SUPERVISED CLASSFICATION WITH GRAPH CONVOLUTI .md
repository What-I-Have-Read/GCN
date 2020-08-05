# SEMI-SUPERVISED CLASSFICATION WITH GRAPH CONVOLUTIONAL NETWORKS

Author: Thmos N.Kipf
Link: https://arxiv.org/pdf/1609.02907.pdf
Publisher: ICLR
Publishing/Release Date: Feb 22, 2017
Score /5: ⭐️⭐️⭐️⭐️⭐️
status: No

# overview

这篇文章是GCN领域的经典论文，使用频谱卷积作用在图上，完成半监督学习任务。问题的提出，假设我们需要完成对一个图上的所以节点进行分类。但是我们只有这个图上的少量节点的信息。

# introduction

文章以这样一个问题开始，假设我们需要对一个引文网络中的文档进行分类这就是一个图问题，我们有一个图，我们需要对图中的节点进行分类，但是我们只有图中一小部分节点的标签。这个问题就是一个基于图的半监督学习问题。


<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathcal{L}_{0}%20+%20\lambda\mathcal{L}_{reg},%20\%20with%20\%20%20\mathcal{L}_{reg}%20=%20\sum_{i,j}A_{ij}\parallel%20f(X_i)%20-%20f(X_j)\parallel^2%20=%20f(X)^T\Delta%20f(X)"/> </p>


这里 ![math](https://latex.codecogs.com/svg.latex?\mathcal{L}_{0}) 代表监督学习部分的loss，即这个图中有标签的节点产生的loss。 ![math](https://latex.codecogs.com/svg.latex?f(.))   一般是一个小的神经网络或者可微函数。 ![math](https://latex.codecogs.com/svg.latex?\lambda)  是一个常量， ![math](https://latex.codecogs.com/svg.latex?X)  是一个包含所有节点特征向量的矩阵。 ![math](https://latex.codecogs.com/svg.latex?\Delta%20=%20D%20-%20A)  表示这个图的拉普拉斯矩阵这，这个图是一个无向图 ![math](https://latex.codecogs.com/svg.latex?\mathcal{G}%20=%20(\mathcal{V},%20\mathcal{E}))  ，它有 ![math](https://latex.codecogs.com/svg.latex?N) 个节点， ![math](https://latex.codecogs.com/svg.latex?V) 个边， ![math](https://latex.codecogs.com/svg.latex?\mathcal{v}_{i}%20\in%20\mathcal{V}) 且 ![math](https://latex.codecogs.com/svg.latex?(v_i,%20v_j)%20\in%20\mathcal{E}) ，还有一个邻接矩阵 ![math](https://latex.codecogs.com/svg.latex?A%20\in%20\sum_j%20A_{i,j}) 和一个度矩阵 ![math](https://latex.codecogs.com/svg.latex?D_{ii}%20=%20\sum_j%20A_{ij}) 。在上述等式的建模方式中，我们假设图中的节点，如果他们相连，那么他们更有可能拥有相同的标签。

针对这个问题，我们首先使用一个神经网络模型 ![math](https://latex.codecogs.com/svg.latex?f(X,A))  对所有带标签的节点进行训练，获得监督学习部分的loss  ![math](https://latex.codecogs.com/svg.latex?\mathcal{L}_0) , 再使用 ![math](https://latex.codecogs.com/svg.latex?f(.)) 对图的邻接矩阵进行处理，这将允许模型从监督损失L0中分配梯度信息，并使其能够学习带标签和不带标签的节点表示。

这篇文章的工作主要有两点，1. 文中针对上述的神经网络模型提出了一个简单，并且表现良好的分层传播机制。并解释了如何从谱图卷积中激发这个想法。2. 文中演示了这种基于图的神经网络模型如何用于快速和可扩展的图中节点的半监督分类。并且在大量数据集上的实验表明，文中提出的模型在分类精度和效率等方面都优于先进的半监督学习方法。

## FAST APPROXIMATE CONVOLUTIONS GRAPHS

文中提出了一个基于如下传播规则的GCN：


<p align="center"> <img src="https://latex.codecogs.com/svg.latex?H^{l+1}%20=%20\delta(\tilde{D}^{-%20\frac{1}{2}}%20\tilde{A}%20\tilde{D}^{-%20\frac{1}{2}}%20H^{l}%20W^{l})"/> </p>


其中， ![math](https://latex.codecogs.com/svg.latex?\tilde{A}%20=%20A%20+%20I_{N})  是一个无向图 ![math](https://latex.codecogs.com/svg.latex?\mathcal{G}) 的邻接矩阵和一个自连接矩阵的和。 ![math](https://latex.codecogs.com/svg.latex?I_{N})  是一个单位矩阵， ![math](https://latex.codecogs.com/svg.latex?\tilde{D}_{ii}%20=%20\sum_{j}\tilde{A}_{ij})  ， ![math](https://latex.codecogs.com/svg.latex?W^{l})  是其中一层的权重矩阵。 ![math](https://latex.codecogs.com/svg.latex?\sigma(.))  是一个激活函数， ![math](https://latex.codecogs.com/svg.latex?H^{l}%20\in%20R^{N\times%20D})  是网络的第 ![math](https://latex.codecogs.com/svg.latex?l) 层参数矩阵， ![math](https://latex.codecogs.com/svg.latex?H^{0}%20=%20X) 。

## SEMI-SUPERVISED NODE CLASSIFICATION

这里我们先跳过文章的证明，回到这个半监督的节点分类问题。文中最终使用两层的GCN网络，在预处理部分，先完成 ![math](https://latex.codecogs.com/svg.latex?\hat{A}%20=%20\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}})  的运算，整个模型的传播过程就是下面这个简单形式：


<p align="center"> <img src="https://latex.codecogs.com/svg.latex?Z%20=%20f(X,%20A)%20=%20softmax\Big(\hat{A}\%20ReLU\Big(\hat{A}XW^{0}\Big)W^{1}\Big)"/> </p>


![gcn_structure.jpg](./img/gcn_structure.jpg)

- (a) 是模型的示意图，输入 ![math](https://latex.codecogs.com/svg.latex?C) 经过一个GCN网络，转换成feature_map  ![math](https://latex.codecogs.com/svg.latex?F) ，最后生成标签 ![math](https://latex.codecogs.com/svg.latex?Y) ，其中黑色的边，在每一层都共享。
- (b) 是1个两层的GCN模型在使用5%的标签的Cora数据集上训练的可视化效果，不同颜色代表不同分类

这里 ![math](https://latex.codecogs.com/svg.latex?W^0%20\in%20R^{C\times%20H}) 是输入层到隐藏层的权重矩阵。 ![math](https://latex.codecogs.com/svg.latex?W^1%20\in%20R^{H%20\times%20F}) 是隐藏层到输出层的权重矩阵，激活函数使用softmax。最后使用所有标签样本的交叉熵loss作为这个半监督学习任务的loss。


<p align="center"> <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20-%20\sum_{l%20\in%20\mathcal{Y}_{L}}\sum_{f=1}^{F}Y_{lf}\ln{Z_{lf}}"/> </p>


 ![math](https://latex.codecogs.com/svg.latex?\mathcal{Y}_L) 是有标签节点的集合，作者训练这个模型时是使用全局梯度下降，使用完整的数据集对每次训练迭代执行批量梯度下降。对 ![math](https://latex.codecogs.com/svg.latex?A) 使用稀疏表示，则内存需求为 ![math](https://latex.codecogs.com/svg.latex?O(\mathcal{\left|\mathcal{E}\right|})) ，需要内存跟边的数量是线性的，如果内存不够，还是建议使用小批量随机梯度下降。