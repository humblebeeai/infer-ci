
# About Confidence Interval:
Validation confidence interval is called through `ci_val()` function.
Confidence interval is calculated by validating model `n_iterations` times randomly choosing data with probabilty of more than 50%, thus skipping about half of the all batches. Then, the mean validation score of the model for each class of the model is calculated. After validation, results are logged into wandb. All arguments in `validate` function are in use, also, some arguments are added to the function to log the results into wandb.

Added Parameters:
- `n_iterations`: 100: # (int, optional) Number of iterations to validate the model,  # (optional) Default value is 100
- `wb_run_name`: # (str) _Name of the wandb run, # (required for wandb logging)_
- `project_name`: # (str) _Name of the wandb project, # (required for wandb logging)_
- `project_entity`: # (str) _Name of the wandb entity, # (required for wandb logging)_

Note: if any of `wb_run_name`, `project_name`, `project_entity` are not provided, the results will only be saved locally. This meesage can also be seen in the terminal at the start of the validation.

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