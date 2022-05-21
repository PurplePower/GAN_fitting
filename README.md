# GAN-based low dimensional data fitting

# 基于GAN的低维数据拟合

该项目实现了多个GAN的变种，并针对二维数据进行拟合及可视化，以便进行分析。

实现的GAN包括：

- GAN, see `Generative Adversarial Networks, Ian Goodfellow et al.`
- Kernel GAN, `Non-parametric estimation of Jensen-Shannon Divergence in 
Generative Adversarial Network training, Mathieu Sinn & Ambrish  Rawat`
- f-GAN, `f -GAN: Training Generative Neural Samplers using 
Variational  Divergence  Minimization, Sebastian Nowozin et al.`
- WGAN-GP, `Wasserstein GAN, Martin Arjovsky et al.` 
and `Improved Training of Wasserstein GAN, Ishaan Gulrajani et al.`
- Least Square GAN, `Least Squares Generative Adversarial Networks, 
Xudong Mao et al.`
- SWG & SWGAN, `Generative Modeling using the Sliced Wasserstein Distance, 
Ishan Deshpande et al.` and `Sliced Wasserstein Generative  Models, Jiqing Wu et al.`

实现主要针对二维数据，尽管部分代码考虑了更高维数据，但正确性尚未测试。
绘图函数也仅支持对二维数据绘图。

## Models

在 `models/` 下实现， 由 `BaseGAN` 定义接口。
构造 GAN 时可以给定 D 和 G 的 `keras.Sequential`模型，或给 `None` 使用默认。
不同GAN变种的构造参数见实现。

`train(...)`方法拟合给定数据集，返回每个迭代轮次的 D 和 G 的损失函数，及给定的 metrics 计算值。

`sampler`参数用于每隔 `sample_interval` 次迭代采样模型的`sample_number`个数据点，
并生成分布图像，保存为指定格式，
见 `visualizers/BaseSampler.py` 和 `data.datamaker.py` 中不同数据集关于 sampler 的实现。

`metrics`是一组衡量模型的指标，如生成分布与真实分布的散度，见 `metrics/`。

## 数据集
`data.data_loader.py`可生成一系列二维数据，及相应的 sampler 来绘图。

## 训练和参数测试

`./hparam test *.py`是不同 GAN 进行参数测试的脚本。
对某一个 GAN 变种，给定各种参数，测试不同组合下的结果，保存模型、图像到 `pics/<model name>/`。
测试支持对某个参数组合重复数次测试以观测稳定性。

`GAN-tuning.py`是测试脚本，可在里面任意测试某个 GAN 对某个数据集的拟合。

`advanced_test/`下是对更复杂数据集的测试。

`./draw_fit.py`对一张黑白图片进行拟合，可能需要小心地调节参数，更多的迭代次数（1w+）
及使用复杂的网络结构。

## 功能测试
`tests/`包含测试脚本来验证一些功能的正确性。

## 拟合结果
SWGAN 使用正交投影和 Stiefel manifold 更新的拟合不稳定；
WGAN、Kernel GAN、SWG（使用随机投影）对复杂数据拟合效果尚可。