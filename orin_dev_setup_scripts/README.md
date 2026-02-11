# How to use Orin nano setup scripts:
 `$./00-setup-jetson-pytorch.sh`  
This will install the Jetson AI Lab community kept versions of pytorch. Only way I've found with out having to jump through hoops for setup

 `$./01-setup-shakespeare-pip-dependencies.sh`  
Used for installing required pip dependencies for using Shakespeare, currently working on setting up requirements.txt

 `$./02-setup-arm64-neovim.sh`  
Install neovim, might be broken so will need to double check can be skipped for now

# Setting up to run Shakespeare Model:
 `$bash data/shakespeare_char/get_dataset.sh`
This will pull the dataset for generating Shakespeare

 `$python3 optimization_and_search/orin_run_experiments.py -c explorations/orin_inf.yaml`
Running Shakespeare prediction model time!



## Resources
- [Setting up Pytorch on Jetson](https://medium.com/@surentharm/setting-up-pytorch-on-nvidia-jetson-nano-the-complete-2025-guide-294a7cf62766)
