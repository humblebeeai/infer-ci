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