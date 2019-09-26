# Learning a Discriminative Feature Network for Semantic Segmentation

**8、Learning a Discriminative Feature Network for Semantic Segmentation**

From a macroscopic perspective, regarding each category of pixels as a whole, inherently considers both intra-class consistency and interclass variation.

It means that the task demands discriminative features.

To this end, we present a novel Discriminative Feature Network \(DFN\) to learn the feature representation which considers both the “**intra-class consistency**” and the “**inter-class distinction**”.

**Smooth Network and Border Network**

**The Smooth Network is designed to address the intra-class inconsistency issue.**

保持类内的一致性

To learn a **robust feature** representation for intra-class consistency, we usually consider **two crucial factors**.

**On the one hand**, we need multi-scale and global context features to encode the local and global information.

**On the other hand**, as multi-scale context is introduced, for a certain scale of thing, the features have different extent of discrimination, some of which may predict a false label.

Therefore, it is necessary to select the discriminative and effective features. Motivated by these two aspects, our Smooth Network is presented based on the **U-shape** \[30, 19, 31, 11, 36\] structure to capture the **multi-scale context information**, with the **global average pooling** \[21, 24, 40, 6\] to capture the **global context**.

Also, we propose a **Channel Attention Block** \(CAB\), which utilizes the **high-level features** to guide the selection of **lowlevel features** stage-by-stage.

**Border Network, on the other hand, tries to differentiate the adjacent patches with similar appearances but different semantic labels.**

保持类间的一致性，就是让边界分割更加明显

Most of the existing approaches \[24, 40, 6, 30\] consider the **semantic segmentation** task as a dense recognition problem, which usually ignores explicitly modeling the inter-class relationship. Consider the example in Figure 1\(d\), if more and more global context is integrated into the classificiation process, the computer case next to the monitor can be easily misclassified as a monitor due to the similar appearance. Thus, it is significant to explicitly involve the semantic boundary to guide the learning of the features. It can amplify the variation of features on both sides. In our Border Network, we integrate semantic boundary loss during the training process to learn the discriminative features to enlarge the “inter-class distinction”.

![Screenshot from 2019-06-19 10-51-43.png](.gitbook/assets/0%20%287%29.png)

**3.1. Smooth network**

在语义分割的任务中，大多数现代方法都将其视为一个密集的预测问题。然而，这种预测在某些部分，特别是大区域和复杂场景的部分，有时会产生不正确的结果，称为类内不一致问题。

**类内不一致问题主要是由于缺乏上下文。**因此，我们引入**全局平均池化**的全局上下文\[24、21、40、6\]。但是，**全局上下文的语义信息比较高，不利于空间信息的恢复**。因此，就像大多数现代方法\[40、6、30\]所做的那样，我们需要**多尺度的接受视图和上下文来进一步优化空间信息**。然而，存在一个问题，即不同尺度的接受性视图产生的特征具有不同程度的识别能力，导致结果不一致。因此，我们需要选择更具识别性的特征来预测某一类别的统一语义标签。

在我们提出的网络中，我们使用**resnet**\[14\]作为基础识别模型。根据特征图的大小，该模型可分为五个阶段。根据我们的观察，不同的阶段有不同的识别能力，导致不同的一致性表现。在较低阶段，网络对较精细的空间信息进行编码，但由于其接受视角小，没有空间语境的引导，因而语义一致性较差。而在高级阶段，由于接受度大，语义一致性强，但预测空间粗糙。总的来说，**较低阶段的空间预测更准确，而较高阶段的语义预测更准确**。在此基础上，结合其优点，提出了**利用高阶一致性来指导低阶优化预测的平滑网络**。

我们发现，在当前流行的语义分割体系结构中，主要有两种风格。

第一种是“主干式”，如pspnet\[40\]、deeplab v3\[6\]。它嵌入**不同尺度的上下文信息**，以提高网络与金字塔空间池化模块\[13\]或Atrous空间金字塔池化模块\[5\]的一致性。

另一种是“编码器-编码器样式”，如refinenet\[19\]、全局卷积网络FCN\[30\]。这种类型的网络利用了不同阶段固有的多尺度语境，但**缺乏具有最强一致性的全局语境**。另外，当网络结合相邻阶段的特征时，只需通过信道对这些特征进行总结。此操作忽略不同阶段的不同一致性。

为了弥补这个缺陷，我们首先嵌入了一个**全局平均池化层**\[24\]，以将U形架构\[27，36\]扩展到一个vshape架构。利用全局平均池层，我们将最强的一致性约束引入到网络中作为指导。此外，为了增强一致性，我们设计了一个通道注意块，如图2（c）所示。本设计结合相邻级的特点，计算出一个信道注意向量3（b）。高阶特征提供了很强的一致性指导，而低阶特征则提供了不同的特征识别信息。这样，信道注意向量可以选择识别特征。

![Screenshot from 2019-06-19 15-11-20.png](.gitbook/assets/1%20%288%29.png)

Figure 3. 通道注意块的示意图。

在（a）中，黄色块表示低阶段的特征，而红色块代表高阶段。我们连接相邻阶段的特征来计算一个权重向量，该权重向量对低阶段的特征映射进行重新加权。较热的颜色代表高重量值。

在（b）中，它是来自stage4频道关注区块的真正关注值矢量。较深的蓝色代表较高的重量值。

**通道注意块**：我们的**通道注意块**（cab）旨在改变每个阶段功能的权重，以增强一致性，如图3所示。在FCN体系结构中，卷积算子输出一个分数图，给出每个像素上每个类的概率。在方程1中，得分图的最终得分仅在所有特征图通道上求和。

![Screenshot from 2019-06-19 14-56-22.png](.gitbook/assets/2%20%282%29.png)

其中x是网络的输出特征。 w代表卷积核。和k∈{1,2,...,K}。 K是频道数量。 D是像素位置的集合。

![APHLATEX311.jpeg](.gitbook/assets/3%20%284%29.jpeg)

其中δ是预测概率。 y是网络的输出。

如等式1和等式2所示，最终的预测标签是具有最高概率的类别。因此，我们假设预测结果是某个块的 y0，而其真实标签是y1。因此,我们可以引入一个参数α改变概率值最高y0 y1,如方程3所示。

![APHLATEX320.jpeg](.gitbook/assets/4%20%285%29.jpeg)

其中y是网络和 α = Sigmoid（x; w）的新预测。基于上述渠道注意块（CAB）的表述，我们可以探索其实际意义。在等式1中，它隐含地表示不同信道的权重是相等的。然而，正如第1节所述，不同阶段的特征具有不同程度的辨识度，导致预测的不一致性。为了获得类内一致性预测，我们应该提取判别性特征并抑制没有区别度的特征。因此，在等式3中，α值应用于特征映射x，其表示使用CAB的特征选择。通过这种设计，我们可以使网络获得阶段性的区分特征，使预测内部一致。

**精细残差块：**特征网络中每个阶段的特征映射都经过精化残差块，如图2（b）所示。该块的第一个组件是一个1X1卷积层。我们用它来统一512个频道的数量。同时，它可以将所有频道的信息结合起来。然后，下面是一个基本的残差块，它可以精化特征映射。此外，这个模块可以增强每个阶段的识别能力，受ResNet架构启发\[14,15\]。

**3.2. 边界网络**

在语义分割任务中，预测与具有相似外观的不同类别容易产生混淆，特别是当它们在空间上相邻时。因此，我们需要扩大功能的区别。在这个动机驱动下，我们采用语义边界来指导特征学习。为了提取准确的语义边界，我们应用语义边界的显式监督，使得网络学习到一个具有强大的跨阶段独特能力的特征。因此，我们提出了一个边界网络来扩大特征的类间差异。它使用显式语义边界监督直接学习语义边界，类似于语义边界检测任务。这使得语义边界两侧的特征可以区分。

如第3.1节所述，特征网络有不同的阶段。低阶段特征具有更详细的信息，而高阶段特征具有更高的语义信息。在我们的工作中，我们需要具有更多语义含义的语义边界。因此，我们设计一个自下而上的边界网络。该网络能够同时从低阶段获得准确的边缘信息，并从高阶获得语义信息，从而消除了一些缺乏语义信息的原始边缘。这样，高阶段的语义信息就可以从低阶段逐步地细化出详细的边缘信息。网络的监督信号是通过传统的图像处理方法，如Canny \[2\]，从语义分割的基础上得到的。

为了补救正负样本的不平衡，我们使用焦点损失\[22\]来监督边界网络的输出，如公式4所示。我们调整焦点损失的参数α和γ以获得更好的性能。

![Screenshot from 2019-06-19 15-49-15.png](.gitbook/assets/5%20%281%29.png)

Pk 是 k类的估计概率，k∈{123，K}。而K 是类标签的最大值。

边界网络主要关注分离边界两侧类别的语义边界。为了提取准确的语义边界，两侧的特征将变得更加可区分。这完全达到了我们的目标，尽可能地使类间差异的特征。

3.3 网络架构

通过平滑的网络和边界网络，我们提出了用于语义分割的判别特征网络，如图2（a）所示。

我们使用预先训练的ResNet \[14\]作为基础网络。在平滑网络中，我们在网络顶部添加全局平均池化层以获得最强的一致性。然后我们利用信道注意块来改变信道的权重以进一步提高一致性。同时，在边界网络中，通过明确的语义边界监督，网络获得准确的语义边界，使双边特征更加清晰。在两个子网络的支持下，类内功能变得更加一致，而类内功能变得更加明显。

对于明确的特征改进，我们使用深度监督来获得更好的性能，并使网络更易于优化。在平滑网络中，我们使用softmax损失来监督每个阶段的上采样输出，不包括全局平均汇聚层，而我们使用焦点损失来监督边界网络的输出。最后，我们使用参数λ来平衡分段损失ls和边界损失lb，如公式7所示。

![Screenshot from 2019-06-19 16-26-10.png](.gitbook/assets/6%20%288%29.png)

**4.1 实现细节**

我们提出的网络基于在ImageNet上预训练的ResNet-101 \[32\]。我们使用FCN4 \[27,36\]作为我们的基本分割框架。

**训练：**我们使用批量为32，动量为0.9，重量衰减为0.0001的小批量随机梯度下降（SGD）\[17\]来训练网络。受到\[5,24\]的启发，我们使用“多元”学习率策略，学习率乘以（1-iter/max iter）功率，功率为0.9，初始学习率4e-3。至于λ，我们经过一系列比较实验后最终使用0.1的值。为了测量我们提出的网络的性能，我们使用平均像素相交（平均IOU）作为度量。

**数据扩充：**我们在训练中使用均值减法和随机水平流量来处理PASCAL VOC 2012和Cityscapes。另外，我们发现对输入图像进行随机缩放非常重要，这明显提高了性能。我们在两个数据集上使用5个尺度的{0.5， 0.75，1，1.5, 1.75}。

