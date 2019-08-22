# Training U-net and generating predictions on test data

## Running on Standard Compute node

Run:

```bash
qsub training_unet_standard_bash_script.pbs 
---


## Running on GPU Compute node

```bash
qsub training_unet_gpu_bash_script.pbs
---

Predicted results of test images can be found in data/membrane/test



