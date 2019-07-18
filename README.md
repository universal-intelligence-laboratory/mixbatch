# mixbatch
The mixbatch act as a interpolate to feature space, to make a soft decition margain to enhance generalization.
And the good part is, you can apply mixbatch on any layer you like, any.

## TODO
auto-adjust alpha of mixup layer:
- with train noise of layer?
- with grad?
- with linear schedule of epoch?



# maya X NNI
无尽之海

# 模型解释能力
综合考虑实现成本和收益，建议按照以下排序做

## 训练噪声指标   
- 最后一层dense的trace可能能够指示最后一层矩阵的变化速度，以及模型到底有没有崩塌，但是我不确定
- 整个模型看成一个函数，其二阶导也就是Hessian和Jacobin矩阵可以指示这个模型解的平滑性，可以用于指标，但是太慢
- moving varience of loss，loss的方差不能显示在tensorboard上吗
- 最后一层输出梯度histrogram，这是最好的也是最有意义的指标

## 卷积核可视化   
- code:  
```
def conv_visualize(model,writer,global_step):
    # 可视化卷积核
    # global_step = epoch*step_per_epoch + step
    for name, param in model.named_parameters():
        if 'conv' in name and 'weight' in name:
            in_channels = param.size()[1]
            out_channels = param.size()[0]   # 输出通道，表示卷积核的个数

            k_w, k_h = param.size()[3], param.size()[2]   # 卷积核的尺寸
            kernel_all = param.view(-1, 1, k_w, k_h)  # 每个通道的卷积核
            kernel_grid = vutils.make_grid(kernel_all, normalize=True, scale_each=True, nrow=in_channels)
            writer.add_image(f'{name}_all', kernel_grid, global_step=global_step) 
```


## ablation study 
- ablation指对一个完整的系统的各个组分进行研究，会有增量和减量两种方法
- 增量法：必须保证加上去的功能相互独立，也就是说如果总准确率等于各个idea带来的增长之和才能用。
- 减量法：如果功能不独立的时候，减量法会比较常见和有用一点。
- 神经学中，一般是在一个已经完成的神经系统中，通过把输入全部设置为0，来disable这个神经元，查看相应结果
- audio sr中我们可以训练好后disable某层，这个时候可以直接衡量结果，也可以通过recovery训练，重新查看结果（我们所做的事是从头训练，可能不太恰当）


## embedding可视化   
- writer.add_embedding(features, metadata=label,global_step=i)  要看最新的emb必须重启tb，tbv1.12不能用
- 每个epoch的embedding写完后，可以进行一定的t-sne步骤从而进行效果的大概分析
- 考虑到对immediate feature的需要，要么就用tensorflow sess.run抓回来，要么就用pytorch或者tensorflow动态图模式，keras很麻烦
- code:  
```
    net.eval()
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)  # get data
            features = net.get_fc2(data)   # 这是一个forward函数的子集，可用于get feature
            writer.add_embedding(features, metadata=target, global_step= global_step)  # write embedding
            break  # no need to write embedding for the whole dataloader
```

## 超平面可视化
- 这个可以做但是计算量比较大，暂时可以放一边
- 目前只有loss-landscape项目