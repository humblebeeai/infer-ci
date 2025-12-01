"""Confidence intervals for object detection metrics"""

from typing import Dict, Union, Tuple, Optional, Any, List
import numpy as np
import sys
from pathlib import Path

# Add the customized ultralytics to path
ultralytics_path = Path(__file__).parent.parent / "ultralytics"
if str(ultralytics_path.parent) not in sys.path:
    sys.path.insert(0, str(ultralytics_path.parent))

try:
    from ultralytics.models.yolo.detect.ci_val import DetectionCIValidator
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    DetectionCIValidator = None


def yolo_map_with_ci(
    model: Any,
    data: str = None,
    n_iterations: int = 100,
    wb_run_name: Optional[str] = None,
    project_name: Optional[str] = None,
    project_entity: Optional[str] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute YOLO mAP50-95 per class with confidence intervals using bootstrap validation.

    This function works exactly like ultralytics model.ci_val() - it validates a YOLO model
    multiple times (n_iterations) by randomly sampling approximately 50% of the validation
    data in each iteration, returning the raw bootstrap samples for mAP metrics.

    Parameters
    ----------
    model : Any
        YOLO model object or path to model weights (e.g., 'yolov8n.pt')
    data : str
        Path to dataset YAML configuration file (e.g., 'coco128.yaml')
    n_iterations : int, optional
        Number of validation iterations for bootstrap, by default 100
    wb_run_name : str, optional
        Weights & Biases run name for logging, by default None
    project_name : str, optional
        Weights & Biases project name, by default None
    project_entity : str, optional
        Weights & Biases entity name, by default None
    **kwargs
        Additional validation arguments (batch, imgsz, conf, iou, device, workers, project, etc.)

    Returns
    -------
    List[Dict[str, float]]
        List of dictionaries, one per iteration. Each dict contains per-class mAP50-95 metrics.
        Keys are in format 'metrics/mAP50-95/{class_name}' and 'metrics/all_classes_mAP'.

    Example
    -------
    >>> from confidenceinterval import yolo_map_with_ci
    >>> from ultralytics import YOLO
    >>>
    >>> # Option 1: Pass model path
    >>> stats_list = yolo_map_with_ci(
    ...     model='yolov8n.pt',
    ...     data='coco128.yaml',
    ...     n_iterations=100,
    ...     batch=32
    ... )
    >>>
    >>> # Option 2: Use YOLO model object (similar to model.ci_val())
    >>> model = YOLO('yolov8n.pt')
    >>> stats_list = yolo_map_with_ci(
    ...     model=model,
    ...     data='coco128.yaml',
    ...     n_iterations=100
    ... )
    >>>
    >>> # Each iteration's results are in stats_list
    >>> print(f"Number of iterations: {len(stats_list)}")
    >>> print(f"Metrics per iteration: {list(stats_list[0].keys())}")

    Notes
    -----
    - Returns raw bootstrap samples (list of dicts), not aggregated statistics
    - CI computation and plotting is done automatically by DetectionCIValidator
    - Results are saved to runs/val/ directory with plots and JSON
    - If WandB parameters provided, results logged to Weights & Biases
    """

    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Customized Ultralytics YOLO is not available. "
            f"Please ensure the 'ultralytics' directory exists at: {ultralytics_path}"
        )

    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
    except ImportError as e:
        raise ImportError(
            f"Failed to import YOLO components: {e}. "
            "Please ensure the customized Ultralytics is properly installed."
        )

    # Load model if path is provided
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    # Prepare validation arguments
    val_args = get_cfg(DEFAULT_CFG)
    val_args.mode = 'val'

    # Set data if provided
    if data:
        val_args.data = data

    # Update with all kwargs
    for key, value in kwargs.items():
        if hasattr(val_args, key):
            setattr(val_args, key, value)

    # Set save_dir to runs folder in current working directory if not specified
    if not hasattr(val_args, 'project') or val_args.project == DEFAULT_CFG.project:
        import os
        val_args.project = os.path.join(os.getcwd(), 'runs')

    # Create validator instance
    validator = DetectionCIValidator(args=val_args)

    # Run CI validation - returns stats_list like model.ci_val()
    stats_list = validator(
        model=model.model,
        n_iterations=n_iterations,
        wb_run_name=wb_run_name,
        project_name=project_name,
        project_entity=project_entity
    )

    return stats_list


def yolo_precision_with_ci(
    model: Any,
    data: str = None,
    n_iterations: int = 100,
    wb_run_name: Optional[str] = None,
    project_name: Optional[str] = None,
    project_entity: Optional[str] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute YOLO precision per class with confidence intervals using bootstrap validation.

    Unlike yolo_map_with_ci which returns mAP metrics, this function returns precision metrics
    for each class across n_iterations of bootstrap validation.

    Parameters
    ----------
    model : Any
        YOLO model object or path to model weights
    data : str
        Path to dataset YAML configuration file
    n_iterations : int, optional
        Number of validation iterations, by default 100
    wb_run_name : str, optional
        Weights & Biases run name, by default None
    project_name : str, optional
        Weights & Biases project name, by default None
    project_entity : str, optional
        Weights & Biases entity name, by default None
    **kwargs
        Additional validation arguments

    Returns
    -------
    List[Dict[str, float]]
        List of dictionaries, one per iteration. Each dict contains per-class precision metrics.
        Keys are in format 'metrics/precision/{class_name}'.

    Example
    -------
    >>> from confidenceinterval import yolo_precision_with_ci
    >>>
    >>> stats_list = yolo_precision_with_ci(
    ...     model='yolov8n.pt',
    ...     data='coco128.yaml',
    ...     n_iterations=100
    ... )
    >>>
    >>> # Calculate mean and CI manually
    >>> import numpy as np
    >>> class_name = list(stats_list[0].keys())[0]
    >>> values = [stats[class_name] for stats in stats_list]
    >>> mean_precision = np.mean(values)
    >>> ci = np.percentile(values, [2.5, 97.5])
    >>> print(f"{class_name}: {mean_precision:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    """

    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Customized Ultralytics YOLO is not available. "
            f"Please ensure the 'ultralytics' directory exists at: {ultralytics_path}"
        )

    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
    except ImportError as e:
        raise ImportError(f"Failed to import YOLO components: {e}")

    # Create custom validator for precision
    class PrecisionCIValidator(DetectionCIValidator):
        def get_stats(self):
            """Returns precision statistics per class."""
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
            self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
            self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
            stats.pop("target_img", None)
            if len(stats) and stats["tp"].any():
                self.metrics.process(**stats)

            # Return precision per class instead of mAP
            precisions = {f'metrics/precision/{self.names[i]}': v for i, v in enumerate(self.metrics.box.p)}
            precisions['metrics/all_classes_precision'] = self.metrics.box.mp  # mean precision
            return precisions

        def plot_confidence_intervals(self):
            """Plot precision confidence intervals (override to use 'Precision' labels)."""
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime

            if not self.wb_run_name:
                local_save = self.local_save
            else:
                if 'ci_val' in self.wb_run_name:
                    local_save = self.wb_run_name
                else:
                    local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            classes = list(self.stats_list[0].keys())

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
            ax.set_xlabel('Mean Precision')
            ax.set_title(f'Mean Precision with Confidence Intervals for Each Class\n{local_save}')

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add confidence interval numbers
            for i, (mean, std_dev) in enumerate(zip(means_list, std_devs_list)):
                ax.text(mean + std_dev + 0.01, i, f'{mean:.2f} ± {std_dev:.2f}', va='center', ha='left', fontsize=9)

            plt.tight_layout()

            # Ensure the directory for saving plots exists
            plot_dir = os.path.join(self.save_dir, 'precision_distributions')
            os.makedirs(plot_dir, exist_ok=True)

            # Save plot to run folder
            plot_path = os.path.join(plot_dir, f'Precision_Confidence_Intervals_{local_save}.png')
            plt.savefig(plot_path)

            # Log the plot to W&B
            if self.wb_run_name and self.project_name and self.project_entity:
                import wandb as wb
                wb.log({"Precision Confidence Intervals/confidence_interval": wb.Image(plot_path)})

        def plot_map_distribution(self):
            """Plot precision distribution (override to use 'Precision' labels)."""
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime

            if not self.wb_run_name:
                local_save = self.local_save
            else:
                if 'ci_val' in self.wb_run_name:
                    local_save = self.wb_run_name
                else:
                    local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            classes = list(self.stats_list[0].keys())

            plot_dir = os.path.join(self.save_dir, 'precision_distributions')
            os.makedirs(plot_dir, exist_ok=True)

            for cls in classes:
                precision_list = [metric[cls] for metric in self.stats_list]

                class_name = cls.split('/')[-1]

                mean_precision = np.mean(precision_list)
                confidence_interval = np.percentile(precision_list, [2.5, 97.5])

                print(f'{class_name} - Mean Precision: {mean_precision}')
                print(f'{class_name} - 95% Confidence Interval for Precision: {confidence_interval}')

                plt.figure(figsize=(10, 6))
                plt.hist(precision_list, bins=30, alpha=0.75, edgecolor='black')
                plt.axvline(x=mean_precision, color='red', linestyle='--', label='Mean Precision')
                plt.axvline(x=confidence_interval[0], color='blue', linestyle='--', label='95% CI Lower Bound')
                plt.axvline(x=confidence_interval[1], color='blue', linestyle='--', label='95% CI Upper Bound')
                plt.xlabel('Precision')
                plt.ylabel('Frequency')
                plt.title(f'Class {class_name} - Bootstrap Precision Distribution with 95% Confidence Interval\n{local_save}')
                plt.legend()

                plt.grid(True, which='both', linestyle='--', linewidth=0.5)

                plot_path = os.path.join(plot_dir, f'{class_name}_precision_{local_save}.png')
                plt.savefig(plot_path)
                plt.close()

                if self.wb_run_name and self.project_name and self.project_entity:
                    import wandb as wb
                    wb.log({f"Precision Confidence Intervals/{class_name}_precision_distribution": wb.Image(plot_path)})

    # Load model
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    # Prepare args
    val_args = get_cfg(DEFAULT_CFG)
    val_args.mode = 'val'
    if data:
        val_args.data = data
    for key, value in kwargs.items():
        if hasattr(val_args, key):
            setattr(val_args, key, value)

    # Set save_dir to runs folder in current working directory if not specified
    if not hasattr(val_args, 'project') or val_args.project == DEFAULT_CFG.project:
        import os
        val_args.project = os.path.join(os.getcwd(), 'runs')

    # Import torch for the custom validator
    import torch

    # Run precision CI validation
    validator = PrecisionCIValidator(args=val_args)
    stats_list = validator(
        model=model.model,
        n_iterations=n_iterations,
        wb_run_name=wb_run_name,
        project_name=project_name,
        project_entity=project_entity
    )

    return stats_list


def yolo_recall_with_ci(
    model: Any,
    data: str = None,
    n_iterations: int = 100,
    wb_run_name: Optional[str] = None,
    project_name: Optional[str] = None,
    project_entity: Optional[str] = None,
    **kwargs
) -> List[Dict[str, float]]:
    """
    Compute YOLO recall per class with confidence intervals using bootstrap validation.

    Unlike yolo_map_with_ci which returns mAP metrics, this function returns recall metrics
    for each class across n_iterations of bootstrap validation.

    Parameters
    ----------
    model : Any
        YOLO model object or path to model weights
    data : str
        Path to dataset YAML configuration file
    n_iterations : int, optional
        Number of validation iterations, by default 100
    wb_run_name : str, optional
        Weights & Biases run name, by default None
    project_name : str, optional
        Weights & Biases project name, by default None
    project_entity : str, optional
        Weights & Biases entity name, by default None
    **kwargs
        Additional validation arguments

    Returns
    -------
    List[Dict[str, float]]
        List of dictionaries, one per iteration. Each dict contains per-class recall metrics.
        Keys are in format 'metrics/recall/{class_name}'.

    Example
    -------
    >>> from confidenceinterval import yolo_recall_with_ci
    >>>
    >>> stats_list = yolo_recall_with_ci(
    ...     model='yolov8n.pt',
    ...     data='coco128.yaml',
    ...     n_iterations=100
    ... )
    >>>
    >>> # Calculate mean and CI manually
    >>> import numpy as np
    >>> class_name = list(stats_list[0].keys())[0]
    >>> values = [stats[class_name] for stats in stats_list]
    >>> mean_recall = np.mean(values)
    >>> ci = np.percentile(values, [2.5, 97.5])
    >>> print(f"{class_name}: {mean_recall:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    """

    if not ULTRALYTICS_AVAILABLE:
        raise ImportError(
            "Customized Ultralytics YOLO is not available. "
            f"Please ensure the 'ultralytics' directory exists at: {ultralytics_path}"
        )

    try:
        from ultralytics import YOLO
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
    except ImportError as e:
        raise ImportError(f"Failed to import YOLO components: {e}")

    # Create custom validator for recall
    class RecallCIValidator(DetectionCIValidator):
        def get_stats(self):
            """Returns recall statistics per class."""
            stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in self.stats.items()}
            self.nt_per_class = np.bincount(stats["target_cls"].astype(int), minlength=self.nc)
            self.nt_per_image = np.bincount(stats["target_img"].astype(int), minlength=self.nc)
            stats.pop("target_img", None)
            if len(stats) and stats["tp"].any():
                self.metrics.process(**stats)

            # Return recall per class instead of mAP
            recalls = {f'metrics/recall/{self.names[i]}': v for i, v in enumerate(self.metrics.box.r)}
            recalls['metrics/all_classes_recall'] = self.metrics.box.mr  # mean recall
            return recalls

        def plot_confidence_intervals(self):
            """Plot recall confidence intervals (override to use 'Recall' labels)."""
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime

            if not self.wb_run_name:
                local_save = self.local_save
            else:
                if 'ci_val' in self.wb_run_name:
                    local_save = self.wb_run_name
                else:
                    local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            classes = list(self.stats_list[0].keys())

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
            ax.set_xlabel('Mean Recall')
            ax.set_title(f'Mean Recall with Confidence Intervals for Each Class\n{local_save}')

            ax.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Add confidence interval numbers
            for i, (mean, std_dev) in enumerate(zip(means_list, std_devs_list)):
                ax.text(mean + std_dev + 0.01, i, f'{mean:.2f} ± {std_dev:.2f}', va='center', ha='left', fontsize=9)

            plt.tight_layout()

            # Ensure the directory for saving plots exists
            plot_dir = os.path.join(self.save_dir, 'recall_distributions')
            os.makedirs(plot_dir, exist_ok=True)

            # Save plot to run folder
            plot_path = os.path.join(plot_dir, f'Recall_Confidence_Intervals_{local_save}.png')
            plt.savefig(plot_path)

            # Log the plot to W&B
            if self.wb_run_name and self.project_name and self.project_entity:
                import wandb as wb
                wb.log({"Recall Confidence Intervals/confidence_interval": wb.Image(plot_path)})

        def plot_map_distribution(self):
            """Plot recall distribution (override to use 'Recall' labels)."""
            import matplotlib.pyplot as plt
            import os
            from datetime import datetime

            if not self.wb_run_name:
                local_save = self.local_save
            else:
                if 'ci_val' in self.wb_run_name:
                    local_save = self.wb_run_name
                else:
                    local_save = self.wb_run_name + f'_ci_val_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

            classes = list(self.stats_list[0].keys())

            plot_dir = os.path.join(self.save_dir, 'recall_distributions')
            os.makedirs(plot_dir, exist_ok=True)

            for cls in classes:
                recall_list = [metric[cls] for metric in self.stats_list]

                class_name = cls.split('/')[-1]

                mean_recall = np.mean(recall_list)
                confidence_interval = np.percentile(recall_list, [2.5, 97.5])

                print(f'{class_name} - Mean Recall: {mean_recall}')
                print(f'{class_name} - 95% Confidence Interval for Recall: {confidence_interval}')

                plt.figure(figsize=(10, 6))
                plt.hist(recall_list, bins=30, alpha=0.75, edgecolor='black')
                plt.axvline(x=mean_recall, color='red', linestyle='--', label='Mean Recall')
                plt.axvline(x=confidence_interval[0], color='blue', linestyle='--', label='95% CI Lower Bound')
                plt.axvline(x=confidence_interval[1], color='blue', linestyle='--', label='95% CI Upper Bound')
                plt.xlabel('Recall')
                plt.ylabel('Frequency')
                plt.title(f'Class {class_name} - Bootstrap Recall Distribution with 95% Confidence Interval\n{local_save}')
                plt.legend()

                plt.grid(True, which='both', linestyle='--', linewidth=0.5)

                plot_path = os.path.join(plot_dir, f'{class_name}_recall_{local_save}.png')
                plt.savefig(plot_path)
                plt.close()

                if self.wb_run_name and self.project_name and self.project_entity:
                    import wandb as wb
                    wb.log({f"Recall Confidence Intervals/{class_name}_recall_distribution": wb.Image(plot_path)})

    # Load model
    if isinstance(model, (str, Path)):
        model = YOLO(model)

    # Prepare args
    val_args = get_cfg(DEFAULT_CFG)
    val_args.mode = 'val'
    if data:
        val_args.data = data
    for key, value in kwargs.items():
        if hasattr(val_args, key):
            setattr(val_args, key, value)

    # Set save_dir to runs folder in current working directory if not specified
    if not hasattr(val_args, 'project') or val_args.project == DEFAULT_CFG.project:
        import os
        val_args.project = os.path.join(os.getcwd(), 'runs')

    # Import torch for the custom validator
    import torch

    # Run recall CI validation
    validator = RecallCIValidator(args=val_args)
    stats_list = validator(
        model=model.model,
        n_iterations=n_iterations,
        wb_run_name=wb_run_name,
        project_name=project_name,
        project_entity=project_entity
    )

    return stats_list


# List of available detection metrics
detection_metrics = [
    'yolo_map_with_ci',
    'yolo_precision_with_ci',
    'yolo_recall_with_ci'
]

# For consistency with other metric modules
detection_conf_methods = ['yolo_bootstrap_percentile']
