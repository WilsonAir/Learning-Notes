# Learning-Based Sampling for Natural Image Matting

**10、Learning-Based Sampling for Natural Image Matting**

前景和背景之间的软过渡中的颜色混合通常用合成方程表示

![APHLATEX010.jpeg](.gitbook/assets/0%20%285%29.jpeg)

其中αi ∈ \[ 0, 1 \]

表示像素i处前景的不透明度。我是原始图像，F和B分别代表前景和背景的未知颜色值。通常，用户输入以trimap的形式提供，其为每个像素提供纯粹前景（即不透明, α = 1），纯背景（即透明，α = 0

\) 或未知不透明度的标签。抠图-算法旨在通过利用已知不透明区域中的像素的颜色来估计未知的不透明度。

