## Code for Further Analysis of Scaling Factor Similarities 

The code template includes three parts: 
i\) logging the scaling factor using Wandb, 
ii\) fetching and storing the data using Wandb Api,
iii) calculating Spearman and Kendall correlation coefficients based on stored data.

### Logging the Scaling Factor using Wandb
Matrix-Level
```python
wandb.log(
    {
        f"scaling_factor/scaling_factor{numid}": torch.mean(scaling_factor).item(),
    },
    step=self.global_step,
)
```
Column-Level
```python
index = 0
for i in self.sampling:
    index += 1
    wandb.log(
        {
            f"scaling_factor/sampling{numid}/{index}": scaling_factor[i].item(),
        },
        step=self.global_step,
    )
```
Specific practices can be found in `./fira_adamw`. Note that to make it easier for wandb to determine current step, change the following code in the `./pre_training_c4/torchrun_main.py`.
```python
# if not layer_wise_flag:
#     optimizer.step()
if not layer_wise_flag:
    optimizer.global_step = global_step
    optimizer.step()
```

### Fetching and Storing the data using Wandb Api
Below is the reference template.
```python
import wandb
import numpy as np

size = xxx # Model size LLaMA 60M, 130M, 350M, 1B
rank = xxx # Low-rank or Full-rank
nums = xxx # Number of matrices, LLaMA 60M - 56, 130M - 84, 350M - 168, 1B - 168

api = wandb.Api()
run = api.run("xxx") # URL to a run

# Matrix-Level
history = run.history(keys=["scaling_factor/scaling_factor" + str(i) for i in range(1, nums + 1)])
scaling_factors_matrix = np.array([[row["scaling_factor/scaling_factor" + str(i)] for _, row in history.iterrows()] for i in range(1, nums + 1)])
np.save(f"scaling_factor_matrix_{size}_{rank}.npy", scaling_factors_matrix)

# Column-Level
history = run.history(keys=[f"scaling_factor/sampling{i}/{j}" for i in range(1, nums + 1) for j in range(1, 101)])
scaling_factors_column = np.array([[row[f"scaling_factor/sampling{i}/{j}"] for _, row in history.iterrows()] for i in range(1, nums + 1) for j in range(1, 101)])
np.save(f"scaling_factor_column_{size}_{rank}.npy", scaling_factors_column)
```

### Calculating Spearman and Kendall Correlation Coefficients
Below is the reference template.
```python
import numpy as np
from scipy.stats import spearmanr, kendalltau

# Load stored data
scaling_factors_matrix_low_rank = np.load("xxx")
scaling_factors_matrix_full_rank = np.load("xxx")

scaling_factors_column_low_rank = np.load("xxx")
scaling_factors_column_full_rank = np.load("xxx")

# Matrix-Level
means_matrix_low_rank = scaling_factors_matrix_low_rank.mean(axis=1)
means_matrix_full_rank = scaling_factors_matrix_full_rank.mean(axis=1)

spearman, p_s = spearmanr(means_matrix_low_rank, means_matrix_full_rank)
kendall, p_k = kendalltau(means_matrix_low_rank, means_matrix_full_rank)

print("Matrix-Level")
print(f"Spearman: {spearman}, P-value: {p_s}, Kendall: {kendall}, P-value: {p_k}")

# Column-Level
means_column_low_rank = scaling_factors_column_low_rank.mean(axis=1)
means_column_full_rank = scaling_factors_column_full_rank.mean(axis=1)

spearman, p_s = spearmanr(means_column_low_rank, means_column_full_rank)
kendall, p_k = kendalltau(means_column_low_rank, means_column_full_rank)

print("Column-Level")
print(f"Spearman: {spearman}, P-value: {p_s}, Kendall: {kendall}, P-value: {p_k}")
```

