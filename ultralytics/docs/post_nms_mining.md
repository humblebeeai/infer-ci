In order to mine hard examples, the `PostNMSLoss` was implemented to get loss from the detections after post-processing, namely non-maximum-suppression (NMS), to avoid getting the losses of detections with very low confidence and unnecessary detections that have no effect on predictions.

The `PostNMSLoss` class can be found in `ultralytics/utils/mining_loss.py`. The class is inherited from `torch.nn.Module`. The first step is to check if predictions or targets (ground truth) are empty. If so, the loss is set to zero. Then, it matches the predictions and targets with the help of the `match_predictions_to_targets` function. It considers matching if a prediction has a target with IoU greater than the IoU threshold called `mining_iou`. If there is no match, the rest are considered unmatched.

After matching, two losses are calculated:
1. BCE loss
2. Bbox loss

Binary cross-entropy loss is calculated for matched predictions with the help of the `torch.nn.BCELoss` function. Also, unmatched predictions and targets are handled separately and added to the BCE loss.

Bbox loss is calculated for matched predictions. The loss is calculated for each bbox coordinate (x1, y1, x2, y2) and summed.

Finally, the losses are summed with weights and the total loss is returned.

### Added arguments for `PostNMSLoss`:
- `mining`: False # (bool) use hard example mining (OHEM) for training
- `mining_iou`: 0.45 # (float) IoU threshold for matching predictions to targets for mining
- `mining_classes`: [1] # (int | list[int], optional) class index for mining
- `mining_matches`: False # (bool) find matching bounding boxes for mining
- `mining_visualize`: True # (bool) visualize mining results
- `mining_vis_num`: 100 # (int) number of mining results to visualize

To use `PostNMSLoss`, `mining = True` and `batch = 1` should be set.

Only certain classes can be mined with the help of the `mining_classes` argument. If not specified, only the class with id 1 is mined. The list of classes that need to be mined can be given as a list of integers. If an integer is given, classes from 0 to the given integer are mined: `range(0, mining_classes)`.

If `mining_matches` is set to `True`, function will only consider images that have the same amount of predictions and targets. This is useful when only mismatching bboxes are needed.

If `mining_visualize` is set to `True`, the mining images will be visualized with only given classes. The number of visualized results can be set with the `mining_vis_num` argument.

`PostNMSLoss` can be used with both `train` and `val` functions. However, using `val` is recommended since the `train` function can slightly change the weights. Visualizations will be saved in the `visualizations` folder in the save directory. Loss results will be saved in the save directory of ultralytics as an `img_losses.txt` file. It will contain the path to the image and losses of each image in the dataset:
```
<path/to/the/image1.jpg> <loss>,
<path/to/the/image2.jpg> <loss>,
...
```

### Example usage:
Reminder: if you want to use `PostNMSLoss` with the `train` function, `batch` size and number of `epochs` should be set to 1.

In code editor:
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.train(data="coco128.yaml", 
                      epochs=1,
                      batch=1,
                      project='loss_example_mining', 
                      name='yolov8n_example_training',
                      conf=0.25,
                      mining=True,
                      mining_iou=0.6,
                      mining_classes=[1, 3])
```
or

Reminder: if you want to use `PostNMSLoss` with the `val` function, `batch` size should be set to 1.
```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.val(data="coco128.yaml",
                    batch=1,
                    project='loss_example_mining',
                    name='yolov8n_example_training',
                    conf=0.25,
                    mining=True,
                    mining_iou=0.6,
                    mining_classes=[1, 3])
```