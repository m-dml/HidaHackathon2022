# HIDA Hackathon 2022 for Energy efficienct modeling

This repo is created from contestants in the Hida Energy Hackathon 
(https://github.com/Helmholtz-AI-Energy/AI-HERO-Energy). Please read the competition repos README first.

### Run this code
To run the code in this repo on the HOREKA environment you don't need to change anything or install any environment.
The conda environment can be executed by anyone with access to this system, so you only need to submit the according
sbatch jobs.

For training the model run `sbatch training_conda.sh` in the `training` folder.
For evaluation run `sbatch forecast_conda.sh` in the `evaluation` folder.

### Install the environment yourself:
We provide an `environment.yaml` file which you can use to create a conda environment to work with this repo.
To install it do `conda env create -f environment.yaml`.
If you do that, don't forget to change the path in the sbatch scripts to point to the executable of that environment!

### What was our approach for this Hackathon?
We started by trying out pure statistical models to lower the energy consumption for training. More precise we were
looking into Auto-Regressions and later Vector Auto Regressions, since they don't really have a training process.
The downside to these models was the higher inference costs. 

That is what brought us to DeepAR models. They work similarly, but being written in GPU-capable libraries they are more
efficient especially when it comes to inference. They are also way more complex and being trained correctly can be
way more accurate. 

Therefore, our final results are using the DeepAR model from pytorch-forecasting 
(https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.models.deepar.DeepAR.html).

We only trained the model for 2 epochs, which was enough for converging. Training costed 155.84KJ and inference on the
complete validation dataset is at 16.22KJ.