from mmcv.runner.hooks import HOOKS,Hook
from mmdet3d.datasets.pipelines.transforms_3d import ObjectSample
from mmdet3d.datasets.dataset_wrappers import CBGSDataset

@HOOKS.register_module()
class PasteStopHook(Hook):
    def __init__(self,stop_epoch=-1):
        self.stop_epoch=stop_epoch
        self.stop_flag=False 
        
    def before_train_epoch(self, runner):
        if self.stop_epoch!=-1 and not self.stop_flag:
            if runner.epoch>=self.stop_epoch:
                assert type(runner.data_loader.dataset)==CBGSDataset
                for i,transform in enumerate(runner.data_loader.dataset.dataset.pipeline.transforms):
                    if type(transform)==ObjectSample:
                        runner.data_loader.dataset.dataset.pipeline.transforms.pop(i)
                        self.stop_flag=True
                        print("ok")
                    
            
            

    
        
        

