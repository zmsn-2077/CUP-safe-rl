# Code

## The source code of the CUP algorithm can be found in Omnisafe, which is able to support vectorized and takes only 3 hours per experiment.

## Repo repository: [OmniSafe](https://github.com/OmniSafeAI/omnisafe)
## Algo directory: [OmniSafe/CUP](https://github.com/OmniSafeAI/omnisafe/blob/main/omnisafe/algorithms/on_policy/first_order/cup.py)

The code for the publication at NeurIPS2022: "Constrained Update Projection Approach to Safe Policy Optimization".

You can simplely run python main.py to use it, with default environment "Swimmer-v3".

If you want to run it in other MuJOCO environment, you can download the repository and modify the code block about argparse section in main function.

In other way, you can run terminal command, for example python main.py --env-id "Ant-v3" , to run it in MuJOCO environment: Ant-v3.

All experiments were implemented in Pytorch 1.7.0 with CUDA 11.0 and conducted on an Ubuntu 20.04.2 LTS (GNU/Linux 5.8.0-59-generic x86 64). We will provide the docker vision as soon as possible.

If you have any problems for our implementation: please contact to: yanglong001@pku.edu.cn or jiamg.ji@gmail.com
