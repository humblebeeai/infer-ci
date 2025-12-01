# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

import wandb as wb
import matplotlib.pyplot as plt
from os.path import join

import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import output_to_target, plot_images

import json
from pathlib import Path
from tqdm import tqdm 

import numpy as np
import torch

from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode
import random
random.seed(42)
from datetime import datetime

class DetectionCIValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from yolo.models.yolo.detect import DetectionValidator

        args = dict(model='yolov8n.pt', data='coco8.yaml')
        validator = DetectionValidator(args=args)
        validator()
        ```
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """Initialize detection model with necessary variables and settings."""
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.nt_per_class = None
        self.nt_per_image = None
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.metrics = DetMetrics(save_dir=self.save_dir, on_plot=self.on_plot)
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.lb = []  # for autolabelling
        self.project_name = None
        self.project_entity = None
        self.wb_run_name = None
        self.stats_list = []
        self.n_iterations = None
        self.local_save = None

    @smart_inference_mode()
    def __call__(self, 
                trainer=None, 
                model=None, 
                n_iterations=None,
                wb_run_name=None,
                project_name=None,
                project_entity=None
                ):
        """Supports validation of a pre-trained model if passed or a model being trained if trainer is passed (trainer
        gets priority).
        """
        self.wb_run_name = wb_run_name
        self.project_name = project_name
        self.project_entity = project_entity
        self.n_iterations = n_iterations

        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu"  # force FP16 val during training
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = 1  # export.py models default to batch-size 1
                LOGGER.info(f"Forcing batch=1 square inference (1,3,{imgsz},{imgsz}) for non-PyTorch models")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding

            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch) ###

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )

        stats_list = []

        for i in tqdm(range(n_iterations)):
            bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))

            self.init_metrics(de_parallel(model))
            self.jdict = []  # empty before each val
            
            for batch_i, batch in enumerate(bar):
                if random.random() < 0.5 and batch_i > 0:
                    continue

                self.run_callbacks("on_val_batch_start")
                self.batch_i = batch_i
                # Preprocess
                with dt[0]:
                    batch = self.preprocess(batch)

                # Inference
                with dt[1]:
                    preds = model(batch["img"], augment=augment)

                # Loss
                with dt[2]:
                    if self.training:
                        self.loss += model.loss(batch, preds)[1]

                # Postprocess
                with dt[3]:
                    preds = self.postprocess(preds)

                self.update_metrics(preds, batch)
               
                self.run_callbacks("on_val_batch_end")
        
            result = self.get_stats()
            stats_list.append(result)
            self.check_stats(stats_list[i])
            # self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()

        if self.training:            
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            LOGGER.info(f"Results: {results}")
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: %.1fms preprocess, %.1fms inference, %.1fms loss, %.1fms postprocess per image"
                % tuple(self.speed.values())
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                from os.path import join
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
        self.stats_list = stats_list
        self.ci_wb()
        LOGGER.info(f"\nResults saved to {colorstr('bold', self.save_dir)}")
        return self.stats_list
    
    def ci_wb(self):

        if self.wb_run_name and self.project_name and self.project_entity:
            run_id = self.get_run_id_from_name()

            if len(run_id):
                wb.init(
                    project=self.project_name,
                    id=run_id,  
                    resume="must", 
                    name=self.wb_run_name,
                    settings=wb.Settings(start_method="fork"),
                    entity = self.project_entity,
                )
            else:
                self.wb_run_name = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                wb.init(
                    project=self.project_name,
                    name=self.wb_run_name,
                    settings=wb.Settings(start_method="fork"),
                    entity = self.project_entity,
                )

        else:
            n1 = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}_"
            # n2 = f"{self.args.model}_"
            n3 = f"iteration{self.n_iterations}_"
            n4 = f"batch{self.args.batch}_"
            n5 = f"imgsz{self.args.imgsz}_"
            n6 = f"{self.args.mode}"
            self.local_save = n1 + n3 + n4 + n5 + n6

        json.dump(self.stats_list, open(join(self.save_dir, 'ci_metrics.json'), 'w'))

        self.plot_confidence_intervals()
        
        self.plot_map_distribution()

    def plot_confidence_intervals(self):

        if not self.wb_run_name:
            local_save = self.local_save
        else:
            if 'ci_val' in self.wb_run_name:
                local_save = self.wb_run_name
            else:
                local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        classes = list(self.stats_list[0].keys())
        # classes.remove('metrics/mAP50-95(B)') # Remove the overall mAP metric
        
        # Collect metrics for each class across iterations
        metrics_dict = {cls: [] for cls in classes}
        for metric in self.stats_list:
            for cls in classes:
                metrics_dict[cls].append(metric[cls])
        
        # Calculate mean and confidence interval for each class
        means = {cls: np.mean(values) for cls, values in metrics_dict.items()}
        std_devs = {cls: 2*np.std(values) for cls, values in metrics_dict.items()}
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        class_names = [cls.split('/')[-1] for cls in classes]
        y_pos = np.arange(len(class_names))
        
        means_list = [means[cls] for cls in classes]
        std_devs_list = [std_devs[cls] for cls in classes]
        
        ax.barh(y_pos, means_list, xerr=std_devs_list, align='center', alpha=0.7, ecolor='black', capsize=10)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(class_names)
        ax.invert_yaxis() 
        ax.set_xlabel('Mean mAP@0.5-0.95')
        ax.set_title(f'Mean mAP@0.5-0.95 with Confidence Intervals for Each Class\n{local_save}')
        
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Add confidence interval numbers
        for i, (mean, std_dev) in enumerate(zip(means_list, std_devs_list)):
            ax.text(mean + std_dev + 0.01, i, f'{mean:.2f} ± {std_dev:.2f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        # Ensure the directory for saving plots exists
        plot_dir = os.path.join(self.save_dir, 'mAP_distributions')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Save plot to run folder
        plot_path = os.path.join(plot_dir, f'Confidence_Intervals_{local_save}.png')
        plt.savefig(plot_path)
        
        # Log the plot to W&B
        if self.wb_run_name and self.project_name and self.project_entity:
            wb.log({"Confidence Intervals/confidence_interval": wb.Image(plot_path)})


    def plot_map_distribution(self):

        if not self.wb_run_name:
            local_save = self.local_save
        else:
            if 'ci_val' in self.wb_run_name:
                local_save = self.wb_run_name
            else:
                local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

        classes = list(self.stats_list[0].keys())
        # classes[-1] = 'metrics/mAP50-95/all' # Rename the overall mAP metric to 'all'

        plot_dir = os.path.join(self.save_dir, 'mAP_distributions')
        os.makedirs(plot_dir, exist_ok=True)
        
        for cls in classes:
            map_list = [metric[cls] for metric in self.stats_list]
            
            class_name = cls.split('/')[-1]
            
            mean_accuracy = np.mean(map_list)
            confidence_interval = np.percentile(map_list, [2.5, 97.5])
            
            print(f'{class_name} - Mean Accuracy: {mean_accuracy}')
            print(f'{class_name} - 95% Confidence Interval for Accuracy: {confidence_interval}')
            
            plt.figure(figsize=(10, 6))
            plt.hist(map_list, bins=30, alpha=0.75, edgecolor='black')
            plt.axvline(x=mean_accuracy, color='red', linestyle='--', label='Mean mAP')
            plt.axvline(x=confidence_interval[0], color='blue', linestyle='--', label='95% CI Lower Bound')
            plt.axvline(x=confidence_interval[1], color='blue', linestyle='--', label='95% CI Upper Bound')
            plt.xlabel('mAP')
            plt.ylabel('Frequency')
            plt.title(f'Class {class_name} - Bootstrap mAP50-95 Distribution with 95% Confidence Interval\n{local_save}')
            plt.legend()
    
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            plot_path = os.path.join(plot_dir, f'{class_name}_{local_save}.png')
            plt.savefig(plot_path)

            if self.wb_run_name and self.project_name and self.project_entity:
                wb.log({f"Confidence Intervals/{class_name}_mAP_distribution": wb.Image(plot_path)})

    def get_run_id_from_name(self):
        # Use the W&B API to retrieve the run ID using the run name
        api = wb.Api()
        runs = api.runs(self.project_name)
        for run in runs:
            if run.name == self.wb_run_name:
                return run.id
        return ''

    def preprocess(self, batch):
        """Preprocesses batch of images for YOLO training."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in ["batch_idx", "cls", "bboxes"]:
            batch[k] = batch[k].to(self.device)

        if self.args.save_hybrid:
            height, width = batch["img"].shape[2:]
            nb = len(batch["img"])
            bboxes = batch["bboxes"] * torch.tensor((width, height, width, height), device=self.device)
            self.lb = (
                [
                    torch.cat([batch["cls"][batch["batch_idx"] == i], bboxes[batch["batch_idx"] == i]], dim=-1)
                    for i in range(nb)
                ]
                if self.args.save_hybrid
                else []
            )  # for autolabelling

        return batch

    def init_metrics(self, model):
        """Initialize evaluation metrics for YOLO."""
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = isinstance(val, str) and "coco" in val and val.endswith(f"{os.sep}val2017.txt")  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(len(model.names)))
        self.args.save_json |= (self.is_coco or self.is_lvis) and not self.training  # run on final val if training COCO
        self.names = model.names
        self.nc = len(model.names)
        self.metrics.names = self.names
        self.metrics.plot = self.args.plots
        self.confusion_matrix = ConfusionMatrix(nc=self.nc, conf=self.args.conf)
        self.seen = 0
        self.jdict = []
        self.stats = dict(tp=[], conf=[], pred_cls=[], target_cls=[], target_img=[])

    def get_desc(self):
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds):
        """Apply Non-maximum suppression to prediction outputs."""

        return ops.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            multi_label=True,
            agnostic=self.args.single_cls,
            max_det=self.args.max_det,
        )

    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred, pbatch):
        """Prepares a batch of images and annotations for validation."""
        predn = pred.clone()
        ops.scale_boxes(
            pbatch["imgsz"], predn[:, :4], pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )  # native-space pred
        return predn

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    self.confusion_matrix.process_batch(predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                file = self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt'
                self.save_one_txt(predn, self.args.save_conf, pbatch["ori_shape"], file)

    def finalize_metrics(self, *args, **kwargs):
        """Set final values for metrics speed and confusion matrix."""
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix

    def get_stats(self):
        """Returns metrics statistics and results dictionary."""
        stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}  # to numpy
        self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
        self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
        stats.pop("target_img", None)
        if len(stats) and stats["tp"].any():
            self.metrics.process(**stats)
        
        maps = {f'metrics/mAP50-95/{self.names[i]}':v for i,v in enumerate(self.metrics.box.maps)}
        maps['metrics/all_classes_mAP'] = self.metrics.results_dict['metrics/mAP50-95(B)']
        return maps

    def print_results(self):
        """Prints training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, names=self.names.values(), normalize=normalize, on_plot=self.on_plot
                )

    def _process_batch(self, detections, gt_bboxes, gt_cls):
        """
        Return correct prediction matrix.

        Args:
            detections (torch.Tensor): Tensor of shape [N, 6] representing detections.
                Each detection is of the format: x1, y1, x2, y2, conf, class.
            labels (torch.Tensor): Tensor of shape [M, 5] representing labels.
                Each label is of the format: class, x1, y1, x2, y2.

        Returns:
            (torch.Tensor): Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        iou = box_iou(gt_bboxes, detections[:, :4])
        return self.match_predictions(detections[:, 5], gt_cls, iou)

    def build_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path, batch_size):
        """Construct and return dataloader."""
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch, ni):
        """Plot validation image samples."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(self, batch, preds, ni):
        """Plots predicted bounding boxes on input images and saves the result."""
        plot_images(
            batch["img"],
            *output_to_target(preds, max_det=self.args.max_det),
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn, save_conf, shape, file):
        """Save YOLO detections to a txt file in normalized coordinates in a specific format."""
        gn = torch.tensor(shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in predn.tolist():
            xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
            with open(file, "a") as f:
                f.write(("%g " * len(line)).rstrip() % line + "\n")

    def pred_to_json(self, predn, filename):
        """Serialize YOLO predictions to COCO json format."""
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn[:, :4])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for p, b in zip(predn.tolist(), box.tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(p[5])]
                    + (1 if self.is_lvis else 0),  # index starts from 1 if it's lvis
                    "bbox": [round(x, 3) for x in b],
                    "score": round(p[4], 5),
                }
            )

    def eval_json(self, stats):
        """Evaluates YOLO output in JSON format and returns performance statistics."""
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            pred_json = self.save_dir / "predictions.json"  # predictions
            anno_json = (
                self.data["path"]
                / "annotations"
                / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
            )  # annotations
            pkg = "pycocotools" if self.is_coco else "lvis"
            LOGGER.info(f"\nEvaluating {pkg} mAP using {pred_json} and {anno_json}...")
            try:  # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                check_requirements("pycocotools>=2.0.6" if self.is_coco else "lvis>=0.5.3")
                if self.is_coco:
                    from pycocotools.coco import COCO  # noqa
                    from pycocotools.cocoeval import COCOeval  # noqa

                    anno = COCO(str(anno_json))  # init annotations api
                    pred = anno.loadRes(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = COCOeval(anno, pred, "bbox")
                else:
                    from lvis import LVIS, LVISEval

                    anno = LVIS(str(anno_json))  # init annotations api
                    pred = anno._load_json(str(pred_json))  # init predictions api (must pass string, not Path)
                    val = LVISEval(anno, pred, "bbox")
                val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                val.evaluate()
                val.accumulate()
                val.summarize()
                if self.is_lvis:
                    val.print_results()  # explicitly call print_results
                # update mAP50-95 and mAP50
                stats[self.metrics.keys[-1]], stats[self.metrics.keys[-2]] = (
                    val.stats[:2] if self.is_coco else [val.results["AP50"], val.results["AP"]]
                )
            except Exception as e:
                LOGGER.warning(f"{pkg} unable to run: {e}")
        return stats
