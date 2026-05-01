# Person Count Classification from Delay-Doppler Data

This [project]([https://duckduckgo.com](https://github.com/Foyceek/MLF_2026_HECL_Frantisek/blob/main/MLF_Project_final.ipynb)) focuses on classifying the number of people in a room using delay-Doppler signal representations. The problem is formulated as an image classification task, where each sample is a 2D snapshot derived from 60 GHz signal reflections.

A convolutional neural network (ResNet-18 with transfer learning) is used to predict one of four classes:
- 0 persons (machine only)
- 1 person
- 2 persons
- 3 persons

## Approach

The pipeline includes:
- Data preprocessing and augmentation
- Optional hyperparameter tuning using Optuna
- Final training with full fine-tuning
- Evaluation using validation accuracy and confusion matrix
- Submission generation for test data

During hyperparameter tuning, the backbone is frozen for efficiency. In the final training phase, the full network is fine-tuned for best performance.

## Configuration

The training behavior is controlled through a central configuration block:

- `BATCH_SIZE`, `EPOCHS`, `LR`, `IMAGE_SIZE` define core training parameters
- `PATIENCE` controls early stopping during final training
- `N_TRIALS` sets the number of Optuna trials for hyperparameter search
- `RUN_HYPERPARAM_TUNING` toggles between using Optuna or predefined parameters

A key feature is the use of `DEFAULT_BEST_PARAMS`, which stores previously optimized hyperparameters. This allows:
- Faster experimentation without rerunning Optuna
- A warm start for further hyperparameter tuning

The code automatically selects GPU if available and enables mixed precision training via `GradScaler` for improved performance.

## Notes

- Set `RUN_HYPERPARAM_TUNING = True` to perform a full search
- Otherwise, the model trains using stored best parameters
- The pipeline is designed to balance efficiency (during tuning) and accuracy (during final training)
