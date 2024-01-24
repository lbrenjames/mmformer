import torch
from predict import AverageMeter, test_softmax
from data.datasets_nii import Brats_loadall_test_nii
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
import mmformer

if __name__ == '__main__':
    masks = [[False, False, False, True], [False, True, False, False], [False, False, True, False], [True, False, False, False],
         [False, True, False, True], [False, True, True, False], [True, False, True, False], [False, False, True, True], [True, False, False, True], [True, True, False, False],
         [True, True, True, False], [True, False, True, True], [True, True, False, True], [False, True, True, True],
         [True, True, True, True]]
    mask_name = ['t2', 't1c', 't1', 'flair', 
            't1cet2', 't1cet1', 'flairt1', 't1t2', 'flairt2', 'flairt1ce',
            'flairt1cet1', 'flairt1t2', 'flairt1cet2', 't1cet1t2',
            'flairt1cet1t2']
    #测试数据的转换方式
    test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
    #change////////
    
    datapath = './BraTS18/BRATS2018_Training_none_npy'
    test_file = 'test2.txt'
    resume = 'output/model_last.pth'
    #changeEnd
    
    #类别的数量
    num_cls = 4
    dataname = 'BRATS2018'
    
    #加载测试数据集
    test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)
    
    #shuffle=False: 这表示在每个epoch开始时，数据加载器不会打乱数据的顺序
    #num_workers=0: 这是用于数据加载的子进程数量。设置为0意味着所有的数据加载将在主进程中进行，不使用额外的子进程。
    #pin_memory=True: 当设置为 True 时，数据加载到GPU前会被固定在内存中，这可以提高数据传输到GPU的速度
    test_loader = MultiEpochsDataLoader(dataset=test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    model = mmformer.Model(num_cls=num_cls)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()
    
    #加载权重模型
    checkpoint = torch.load(resume)
    model.load_state_dict(checkpoint['state_dict'])

    test_score = AverageMeter()
    with torch.no_grad():
        print('###########test set wi/wo postprocess###########')
        for i, mask in enumerate(masks):
            print('{}'.format(mask_name[i]))
            dice_score = test_softmax(
                            test_loader,
                            model,
                            dataname = dataname,
                            feature_mask = mask)
            test_score.update(dice_score)
        print('Avg scores: {}'.format(test_score.avg))
