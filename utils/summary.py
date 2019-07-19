import numpy as np
import torchvision.utils as vutils

def mean_std(loss_list,writer,global_step,name="loss"):
    min_total = 20
    max_elements = 30
    min_elements = int(min_total*0.3)
    data_len = len(loss_list)

    if data_len >= min_total:
        mean_len = int(data_len*0.3)
        mean_len = min(mean_len,max_elements)
        data = loss_list[-1*mean_len:]
        mean = np.mean(data)
        std = np.std(data)
        writer.add_scalar(name+'_mean', mean, global_step)
        writer.add_scalar(name+'_std', std, global_step)


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
