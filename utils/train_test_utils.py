from detectron2.structures import BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.engine.hooks import HookBase
import detectron2.utils.comm as comm
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader, DatasetMapper
from detectron2.data import detection_utils
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator


import numpy as np
from datetime import datetime
import os, copy, torch,time


MODEL_CONFIGS_LINKS = {'HJDataset': 'https://www.dropbox.com/s/j4yseny2u0hn22r/config.yml?dl=1', 
                'MFD': 'https://www.dropbox.com/s/ld9izb95f19369w/config.yaml?dl=1',
                'NewspaperNavigator': 'https://www.dropbox.com/s/wnido8pk4oubyzr/config.yml?dl=1', 
                'PubLayNet': 'https://www.dropbox.com/s/f3b12qc4hc0yh4m/config.yml?dl=1', 
                 'TableBank': 'https://www.dropbox.com/s/7cqle02do7ah7k4/config.yaml?dl=1'}


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader
    
    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)
            
        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):            
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                
                # log_every_n_seconds(
                #     logging.INFO,
                #     "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                #         idx + 1, total, seconds_per_img, str(eta)
                #     ),
                #     n=5,
                # )
            
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)

        mean_loss = np.mean(losses)
        print(f"Mean Val Loss: {mean_loss}")
        self.trainer.storage.put_scalar('validation_loss', mean_loss)
        comm.synchronize()
        
        return losses
            
    def _get_loss(self, data):
        # How loss is calculated on train_loop 
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced
        
        
    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)




def custom_mapper(dataset_dict, transform_list = None):
    
    if transform_list is None:
      transform_list = [T.RandomBrightness(0.8, 1.2),
                      T.RandomContrast(0.8, 1.2),
                      T.RandomSaturation(0.8, 1.2),
                      ]
                      
    dataset_dict = copy.deepcopy(dataset_dict)
    image = detection_utils.read_image(dataset_dict["file_name"], format="BGR")
                      
    image, transforms = T.apply_transform_gens(transform_list, image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

    annos = [
        detection_utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = detection_utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = detection_utils.filter_empty_instances(instances)
    return dataset_dict


class AugTrainer(DefaultTrainer): # Trainer with augmentations
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=custom_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR,"inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1,LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        ))
        return hooks


def download_config(url, name):
    import subprocess
    name = f"./{name}_config.yml"
    subprocess.Popen(f"""wget -qq -O {name} {url}""", shell=True)
    time.sleep(3)
    return name



def build_layoutParser_faster_rcnnn_config(pretrained_model_name:str, Data_Resister_training:str, Data_Resister_valid:str):
    '''
    Build and Modify the parameters for the config from the Layout Parser models
    args:
        pretrained_model_name: Name of the pretrained model from Layout Parser whose config you want to use ['HJDataset', 'MFD', 'NewspaperNavigator', 'PubLayNet', 'TableBank']
    '''
    assert pretrained_model_name in ['HJDataset', 'MFD', 'NewspaperNavigator', 'PubLayNet', 'TableBank'], "'pretrained_weights' must be one of ['HJDataset', 'MFD', 'NewspaperNavigator', 'PubLayNet', 'TableBank']"
    
    config_name = download_config(MODEL_CONFIGS_LINKS[pretrained_model_name], pretrained_model_name)

    cfg = get_cfg()

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    cfg.merge_from_file(config_name)

    cfg.MODEL.DEVICE = "cpu" if not torch.cuda.is_available() else "cuda"

    # cfg.CUDNN_BENCHMARK = True #  Default False
    cfg.DATALOADER.NUM_WORKERS: 2

    cfg.DATASETS.TRAIN = (Data_Resister_training,)
    cfg.DATASETS.TEST = (Data_Resister_valid,)

    cfg.TEST.EVAL_PERIOD = 20 # Evaluate after N epochs

    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128 # Default 256 . More the number, more the memory consumption
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # in config file, it is written before weights

    # If using config.ym file for any pre trained model, weights path is already there so no need to download weights
    # cfg.MODEL.WEIGHTS= './faster_rcnn_R_50_FPN_3x.pth' # layout parser Pre trained weights 

    cfg.MODEL.MASK_ON = False # In case we have used Mask RCNN model


    cfg.SOLVER.IMS_PER_BATCH = 4 # Batch size
    cfg.SOLVER.BASE_LR = 0.0025
    cfg.SOLVER.WARMUP_ITERS = 50
    cfg.SOLVER.MAX_ITER = 1000 # adjust up if val mAP is still rising, adjust down if overfit
    cfg.SOLVER.STEPS = (300, 800) # must be less than  MAX_ITER 
    cfg.SOLVER.GAMMA = 0.05
    cfg.SOLVER.CHECKPOINT_PERIOD = 20  # Save weights after these many steps

    return cfg



def build_test_model(cfg, model_weights_path:str, thresh:float = 0.6):
    '''
    build a predictor model given loaded config, model weights and a threshold for NMS
    '''
    cfg.MODEL.WEIGHTS = model_weights_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = thresh
    predictor = DefaultPredictor(cfg)
    return predictor
