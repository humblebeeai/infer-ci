import torch
import torch.nn as nn
import os
import cv2

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.metrics import bbox_iou

class BboxLoss(nn.Module):
    """Criterion class for computing training losses during training."""
    def __init__(self):
        super().__init__()

    def forward(self, pred_bboxes, target_bboxes, target_scores):
        """IoU loss."""
        iou = bbox_iou(pred_bboxes, target_bboxes, xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou)).sum() / target_scores.size(0)
        return loss_iou

class PostNMSLoss(nn.Module):
    def __init__(self):    
        super(PostNMSLoss, self).__init__()
        self.iou_threshold = 0.45
        self.cls_loss_fn = nn.BCELoss()
        self.bbox_loss_fn = BboxLoss()
        self.device = None
        self.visualize_dir = "visualizations"
        self.count = 0
        self.classes = {}

    def forward(self, obj, preds, batch):
        """
        Args:
            preds: Tensor of shape [N, 6], where N is the number of detections after NMS.
                   Format: [x1, y1, x2, y2, confidence, class_id]
            batch: Dictionary containing ground truth data.
        
        Returns:
            total_loss: Combined loss value (scalar).
        """
        nc = obj.nc
        self.iou_threshold = obj.args.mining_iou
        self.classes = obj.names
        self.device = obj.device
        self.visualize_dir = obj.save_dir
        hyp_cls = obj.args.cls
        hyp_bbox = obj.args.box
        batch_size = len(preds)
        preds = preds[0]

        # Filter for mining classes
        mining_classes = (
            obj.args.mining_classes
            if isinstance(obj.args.mining_classes, list)
            else range(obj.args.mining_classes)
            if isinstance(obj.args.mining_classes, int)
            else obj.class_map
        )
        class_mask = torch.isin(preds[:, 5], torch.tensor(mining_classes, device=self.device))
        preds = preds[class_mask]

        if preds.size(0) == 0:  # No detections for class 1
            return torch.tensor(0.0)

        # Separate predictions into components
        pred_boxes = preds[:, :4]   # Bounding box predictions
        pred_scores = preds[:, 4]   # Confidence scores

        dtype = pred_scores.dtype
        imgsz = torch.tensor(batch["resized_shape"][0], device=self.device, dtype=dtype) # image size (h,w)

        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = self.preprocess(targets.to(self.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])[0]

        gt_labels, gt_bboxes = targets.split((1, 4), 1)  # cls, xyxy
        targets = torch.cat((gt_bboxes, gt_labels), 1)

        # Filter targets for class given in mining_classes
        class_mask = torch.isin(targets[:, 4], torch.tensor(mining_classes, device=preds.device))
        targets = targets[class_mask]


        if targets.size(0) == 0:  # No targets for class 1
            return torch.tensor(0.0)

        # Separate targets into components
        target_boxes = targets[:, :4]
        target_scores = targets[:, 4]

        # Match predictions to targets
        matched_preds, matched_targets, unmatched_preds, unmatched_targets = self.match_predictions_to_targets(preds, targets)

        if matched_preds.size(0) == 0:  # No matched predictions
            return torch.tensor(0.0)

        if obj.args.mining_matches:
            if preds.shape[0] != targets.shape[0] and unmatched_preds.size(0) > 0:
                return torch.tensor(0.0)
        
        pred_class_ids = matched_preds[:, 5].long() # Predicted class IDs
        target_class_ids = matched_targets[:, 4].long()  # Ground truth class IDs

        pred_boxes = matched_preds[:, :4]
        pred_scores = matched_preds[:, 4]
        target_boxes = matched_targets[:, :4]
        target_scores = matched_targets[:, 4]

        n_matches = int(matched_preds.size(0))

        # Convert target class IDs to one-hot encoding
        bce_targets = torch.zeros(target_class_ids.size(0), nc, device=self.device)
        bce_preds = torch.zeros(len(pred_class_ids), nc, device=self.device)

        for i, cls in enumerate(target_class_ids):
            bce_targets[i, cls] = 1
        for i, cls in enumerate(pred_class_ids):
            bce_preds[i, cls] = pred_scores[i]

        # Calculate classification loss (BCE)
        cls_loss = self.cls_loss_fn(bce_preds, bce_targets)

        # Add log(pred_score) for unmatched predictions
        if unmatched_preds.size(0) > 0:
            cls_loss -= unmatched_preds[:, 4].log().sum()

        # Add 1 for unmatched targets
        if unmatched_targets.size(0) > 0:
            cls_loss += unmatched_targets.size(0)

        # Calculate bounding box loss
        bbox_loss = self.bbox_loss_fn(pred_boxes, target_boxes, target_scores)

        # Combine losses with weights
        total_loss = hyp_cls * cls_loss + hyp_bbox * bbox_loss

        if self.count < obj.args.mining_vis_num and obj.args.mining_visualize:
            self.visualize(preds, targets, batch, total_loss, bbox_loss, cls_loss, n_matches)
            self.count += 1

        return total_loss
    
    def visualize(self, preds, targets, batch, total_loss, bbox_loss, cls_loss, n_matches):
        """Visualize the predictions and targets."""
        import numpy as np


        image_tensor = batch["img"] # normalized tensor image
        augmented_image = image_tensor[0].permute(1, 2, 0).cpu().numpy() * 255 # numpy array image with values between 0 and 255
        if augmented_image.dtype == np.float16:  # Check for CV_16F
            augmented_image = augmented_image.astype(np.float32)

        image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR) # convert the image from numpy array to OpenCV format

        preds = preds.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        cv2.putText(image, f"Preds: {int(preds.shape[0])}, Targets: {int(targets.shape[0])}, Matches: {n_matches}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        for bbox, score, label in zip(preds[:, :4], preds[:, 4], preds[:, 5]):

            # bbox format is [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = bbox[:4]
            color = (0, 0, 255) # Red color

            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(image, f"{self.classes[int(label)]}  {score:.3f}", (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

            # Define the text to be written
            text = f'Loss --> box:{bbox_loss:.2f}, cls:{cls_loss:.2f}, total:{total_loss:.2f}'

            # Get image dimensions
            height, width, _ = image.shape

            # Define the position where the text will be written (bottom of the image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            color = (255, 255, 255)  # White color
            thickness = 2

            # Calculate text size
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]

            # Set the position to bottom center
            text_x = (width - text_size[0]) // 2
            text_y = height - 10  # A little above the bottom edge

            # Put the text on the image
            cv2.putText(image, text, (text_x, text_y), font, font_scale, color, thickness)

        for bbox, label in zip(targets[:, :4], targets[:, 4]):
            color = (0, 255, 0)
            x_min, y_min, x_max, y_max = bbox[:4]

            cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color, 2)
            cv2.putText(image, f"{self.classes[int(label)]}", (int(x_min), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Save the image to a separate file for each key
        img_name = batch["im_file"][0].split("/")[-3:]
        img_name = img_name[0] + "_" + img_name[2]
        self.visualize_dir = os.path.join(self.visualize_dir, "visualizations")
        if not os.path.exists(self.visualize_dir):
            os.makedirs(self.visualize_dir)
        save_path = os.path.join(self.visualize_dir, img_name)
        cv2.imwrite(save_path, image)

    def preprocess(self, targets, batch_size, scale_tensor):
        """Preprocesses the target counts and matches with the input batch size to output a tensor."""
        nl, ne = targets.shape
        if nl == 0:
            out = torch.zeros(batch_size, 0, ne - 1, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(batch_size, counts.max(), ne - 1, device=self.device)
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
        return out
    
    def match_predictions_to_targets(self, preds, targets):
        if preds.size(0) == 0 or targets.size(0) == 0:
            return torch.empty((0, 6), device=self.device), torch.empty((0, 6), device=self.device)

        ious = self.box_iou(preds[:, :4], targets[:, :4])  # Vectorized IoU computation
        max_ious, indices = ious.max(dim=1)  # Get the best matches
        keep = max_ious > self.iou_threshold

        matched_preds = preds[keep]
        matched_targets = targets[indices[keep]]

        # Add unmatched predictions
        unmatched_preds = preds[~keep]
        unmatched_target_mask = torch.ones(targets.size(0), device=self.device, dtype=torch.bool)
        unmatched_target_mask[indices[keep]] = False
        unmatched_targets = targets[unmatched_target_mask]

        return matched_preds, matched_targets, unmatched_preds, unmatched_targets
    
    def box_iou(self, box1, box2, eps=1e-7):
        """
        Calculate intersection-over-union (IoU) of boxes. Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Based on https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py.

        Args:
            box1 (torch.Tensor): A tensor of shape (N, 4) representing N bounding boxes.
            box2 (torch.Tensor): A tensor of shape (M, 4) representing M bounding boxes.
            eps (float, optional): A small value to avoid division by zero. Defaults to 1e-7.

        Returns:
            (torch.Tensor): An NxM tensor containing the pairwise IoU values for every element in box1 and box2.
        """
        # NOTE: Need .float() to get accurate iou values
        # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
        (a1, a2), (b1, b2) = box1.float().unsqueeze(1).chunk(2, 2), box2.float().unsqueeze(0).chunk(2, 2)
        inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp_(0).prod(2)

        # IoU = inter / (area1 + area2 - inter)
        return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)
