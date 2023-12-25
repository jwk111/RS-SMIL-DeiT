# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/home/jwk/Project/remoteSensing/datasets/UCMerced_LandUse/Images' # 数据集的根目录
    model = 'DeiT_small' # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    # MIL_DeiT, DeiT_small
    freeze = True # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 12 # DataLoader 中的多线程数量
    num_classes = 21 # 分类类别数
    num_epochs = 80
    batch_size = 32
    lr = 0.05 # 初始lr
    lrf = 0.00005
    width = 256 # 输入图像的宽
    height = 256 # 输入图像的高
    iter_smooth = 10 # 打印&记录log的频率
    resume = False #
    save_model = 'DeiT_small_UCM6_4_80epoch'
    checkpoint = 'DeiT_small_UCM6_4_80epoch.pth' # 评估使用的模型名

config = DefaultConfigs()


