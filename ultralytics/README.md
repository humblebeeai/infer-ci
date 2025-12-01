# About mAP50-95 per class wandb logging:
mAP50-95 is also logged into wandb during validation for each class. It can be seen under the `mAP50-95` section in the wandb dashboard.
In this section, curve for mAP50-95 for each class is logged 

# About per class loss wandb logging:
Each loss (_box loss, cls loss, dfl loss_) for each class are logged into wandb during training and validation. It can be seen under the following sections in the wandb dashboard:
- `train_box_loss`
- `train_cls_loss`
- `train_dfl_loss`

<br>

- `val_box_loss`
- `val_cls_loss`
- `val_dfl_loss`

In each section, corresponding loss curve for each class is logged.

# About Confidence Interval:
Confidence interval is calculated by validating model `n_iterations` times randomly choosing about 50% of data by skipping some batches, then, calculating the mean validation score of the model for each class of the model. After validation, results are logged into wandb. All arguments in `validate` function are in use, also, some arguments are added to the function to log the results into wandb.

Added Parameters:
- `n_iterations` = 100: Number of iterations to validate the model,  # (optional) Default value is 100
- `wb_run_name`: _Name of the wandb run, # (required for wandb logging)_
- `project_name`: _Name of the wandb project, # (required for wandb logging)_
- `project_entity`: _Name of the wandb entity, # (required for wandb logging)_

Note: if any of `wb_run_name`, `project_name`, `project_entity` are not provided, the results will only be saved locally.

Example Usage:
In terminal:
```bash
wandb login <your_wandb_api_key>
```
In code editor:
```
from ultralytics import YOLO
import wandb

model = YOLO("yolov8n.pt")

results = model.ci_val(
                      data="coco128.yaml",  
                      batch=32,
                      project="my_experiments",
                      n_iterations=100,
                      wb_run_name='YOLOv8n_example_training', 
                      project_name='my_experiments', 
                      project_entity='my_team',
                      )
```