# SEMI-SUPERVISED CLASSFICATION WITH GRAPH CONVOLUTIONAL NETWORKS

Author: Thmos N.Kipf  
Link: https://arxiv.org/pdf/1609.02907.pdf  
Publisher: ICLR  
Publishing/Release Date: Feb 22, 2017  
Score /5: ⭐️⭐️⭐️⭐️⭐️  

# overview

这篇文章是GCN领域的经典论文，使用频谱卷积作用在图上，完成半监督学习任务。问题的提出，假设我们需要完成对一个图上的所以节点进行分类。但是我们只有这个图上的少量节点的信息。

# introduction

文章以这样一个问题开始，假设我们需要对一个引文网络中的文档进行分类这就是一个图问题，我们有一个图，我们需要对图中的节点进行分类，但是我们只有图中一小部分节点的标签。这个问题就是一个基于图的半监督学习问题。

<p align="center">
    <img src="https://latex.codecogs.com/svg.latex?\mathcal{L}%20=%20\mathcal{L}_{0}%20+%20\lambda\mathcal{L}_{reg},%20\%20with%20\%20%20\mathcal{L}_{reg}%20=%20\sum_{i,j}A_{ij}\parallel%20f(X_i)%20-%20f(X_j)\parallel^2%20=%20f(X)^T\Delta%20f(X)"/>
</p>

这里 ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D_%7B0%7D) 代表监督学习部分的loss，即这个图中有标签的节点产生的loss。 ![math](https://render.githubusercontent.com/render/math?math=f%28.%29)   一般是一个小的神经网络或者可微函数。 ![math](https://render.githubusercontent.com/render/math?math=%5Clambda)  是一个常量， ![math](https://render.githubusercontent.com/render/math?math=X)  是一个包含所有节点特征向量的矩阵。 ![math](https://render.githubusercontent.com/render/math?math=%5CDelta%20%3D%20D%20-%20A)  表示这个图的拉普拉斯矩阵这，这个图是一个无向图 ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BG%7D%20%3D%20%28%5Cmathcal%7BV%7D%2C%20%5Cmathcal%7BE%7D%29)  ，它有 ![math](https://render.githubusercontent.com/render/math?math=N) 个节点， ![math](https://render.githubusercontent.com/render/math?math=V) 个边， ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7Bv%7D_%7Bi%7D%20%5Cin%20%5Cmathcal%7BV%7D) 且 ![math](https://render.githubusercontent.com/render/math?math=%28v_i%2C%20v_j%29%20%5Cin%20%5Cmathcal%7BE%7D) ，还有一个邻接矩阵 ![math](https://render.githubusercontent.com/render/math?math=A%20%5Cin%20%5Csum_j%20A_%7Bi%2Cj%7D) 和一个度矩阵 ![math](https://render.githubusercontent.com/render/math?math=D_%7Bii%7D%20%3D%20%5Csum_j%20A_%7Bij%7D) 。在上述等式的建模方式中，我们假设图中的节点，如果他们相连，那么他们更有可能拥有相同的标签。

针对这个问题，我们首先使用一个神经网络模型 ![math](https://render.githubusercontent.com/render/math?math=f%28X%2CA%29)  对所有带标签的节点进行训练，获得监督学习部分的loss  ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D_0) , 再使用 ![math](https://render.githubusercontent.com/render/math?math=f%28.%29) 对图的邻接矩阵进行处理，这将允许模型从监督损失L0中分配梯度信息，并使其能够学习带标签和不带标签的节点表示。

这篇文章的工作主要有两点，1. 文中针对上述的神经网络模型提出了一个简单，并且表现良好的分层传播机制。并解释了如何从谱图卷积中激发这个想法。2. 文中演示了这种基于图的神经网络模型如何用于快速和可扩展的图中节点的半监督分类。并且在大量数据集上的实验表明，文中提出的模型在分类精度和效率等方面都优于先进的半监督学习方法。

## FAST APPROXIMATE CONVOLUTIONS GRAPHS

文中提出了一个基于如下传播规则的GCN：



![math](https://render.githubusercontent.com/render/math?math=%E2%81%8D)



其中， ![math](https://render.githubusercontent.com/render/math?math=%5Ctilde%7BA%7D%20%3D%20A%20%2B%20I_%7BN%7D)  是一个无向图 ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BG%7D) 的邻接矩阵和一个自连接矩阵的和。 ![math](https://render.githubusercontent.com/render/math?math=I_%7BN%7D)  是一个单位矩阵， ![math](https://render.githubusercontent.com/render/math?math=%5Ctilde%7BD%7D_%7Bii%7D%20%3D%20%5Csum_%7Bj%7D%5Ctilde%7BA%7D_%7Bij%7D)  ， ![math](https://render.githubusercontent.com/render/math?math=W%5E%7Bl%7D)  是其中一层的权重矩阵。 ![math](https://render.githubusercontent.com/render/math?math=%5Csigma%28.%29)  是一个激活函数， ![math](https://render.githubusercontent.com/render/math?math=H%5E%7Bl%7D%20%5Cin%20R%5E%7BN%5Ctimes%20D%7D)  是网络的第 ![math](https://render.githubusercontent.com/render/math?math=l) 层参数矩阵， ![math](https://render.githubusercontent.com/render/math?math=H%5E%7B0%7D%20%3D%20X) 。

## SEMI-SUPERVISED NODE CLASSIFICATION

这里我们先跳过文章的证明，回到这个半监督的节点分类问题。文中最终使用两层的GCN网络，在预处理部分，先完成 ![math](https://render.githubusercontent.com/render/math?math=%5Chat%7BA%7D%20%3D%20%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D%5Ctilde%7BA%7D%5Ctilde%7BD%7D%5E%7B-%5Cfrac%7B1%7D%7B2%7D%7D)  的运算，整个模型的传播过程就是下面这个简单形式：



![math](https://render.githubusercontent.com/render/math?math=Z%20%3D%20f%28X%2C%20A%29%20%3D%20softmax%5CBig%28%5Chat%7BA%7D%5C%20ReLU%5CBig%28%5Chat%7BA%7DXW%5E%7B0%7D%5CBig%29W%5E%7B1%7D%5CBig%29)



![gcn_structure.jpg](./img/gcn_structure.jpg)

- (a) 是模型的示意图，输入 ![math](https://render.githubusercontent.com/render/math?math=C) 经过一个GCN网络，转换成feature_map  ![math](https://render.githubusercontent.com/render/math?math=F) ，最后生成标签 ![math](https://render.githubusercontent.com/render/math?math=Y) ，其中黑色的边，在每一层都共享。
- (b) 是1个两层的GCN模型在使用5%的标签的Cora数据集上训练的可视化效果，不同颜色代表不同分类

这里 ![math](https://render.githubusercontent.com/render/math?math=W%5E0%20%5Cin%20R%5E%7BC%5Ctimes%20H%7D) 是输入层到隐藏层的权重矩阵。 ![math](https://render.githubusercontent.com/render/math?math=W%5E1%20%5Cin%20R%5E%7BH%20%5Ctimes%20F%7D) 是隐藏层到输出层的权重矩阵，激活函数使用softmax。最后使用所有标签样本的交叉熵loss作为这个半监督学习任务的loss。



![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BL%7D%20%3D%20-%20%5Csum_%7Bl%20%5Cin%20%5Cmathcal%7BY%7D_%7BL%7D%7D%5Csum_%7Bf%3D1%7D%5E%7BF%7DY_%7Blf%7D%5Cln%7BZ_%7Blf%7D%7D)



 ![math](https://render.githubusercontent.com/render/math?math=%5Cmathcal%7BY%7D_L) 是有标签节点的集合，作者训练这个模型时是使用全局梯度下降，使用完整的数据集对每次训练迭代执行批量梯度下降。对 ![math](https://render.githubusercontent.com/render/math?math=A) 使用稀疏表示，则内存需求为 ![math](https://latex.codecogs.com/svg.latex?O(\mathcal{\left|\mathcal{E}\right|})) ，需要内存跟边的数量是线性的，如果内存不够，还是建议使用小批量随机梯度下降。