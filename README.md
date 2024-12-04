# Safe Learning of Locomotion Skills from MPC
Author: Xun Pua

This repository is the code base for the publication [Safe Learning of Locomotion Skills from MPC](https://arxiv.org/html/2407.11673v1)

## Getting Started
This project is developed using the *devcontainer* function of VSCode. To run this workspace, please first ensure that VSCode is install on your PC and the following Extensions are installed:

- Dev Containers (ms-vscode-remote.remote-containers)
- Remote Development (ms-vscode-remote.vscode-remote-extensionpack)
- Docker (ms-azuretools.vscode-docker)

Also, this project uses [weights and biases (wandb)](https://wandb.ai/site) as the online database. Make sure you create an account for it before running some scripts, as it will prompt you to enter your credentials when you freshly restarted the devcontainer.

### Cloning the Repository
First, clone this repository onto your development PC.
```
git clone git@github.com:XunPua/quadrupedal_iterative_supervised_learning.git
```
OR
```
git clone https://github.com/XunPua/quadrupedal_iterative_supervised_learning.git
```
Then, enter the directory downloaded and open it in vscode
```
cd quadrupedal_iterative_supervised_learning && code .
```

### Starting the devcontainer
in the .devcontainer file, you will find the following files:

|File|Usage|
|---|---|
|devcontainer.local.json | devcontainer for local development|
|devcontainer.remote.json | devcontainer for workstation development|
|devcontainer.json| devcontainer that will be built|
|build_biconvex_mpc.sh| build script for BiConMP (example. not used)|
|build_packages.sh| build script for bullet_utils, robot properties and the BiConMP imbedded in the *iterative_supervised_learning* directory|
|bashrc|all relevant paths for devcontainer|
|Dockerfile|the docker file :)|

you will first need to make sure that the devcontainer.json is the working environment you want (local or workstation). If not, copy a version of the .local. or .remote. json file and rename it to devcontainer.json. 

Before you build, do note that there might be an issue of memory overload in certain cases when building the packages with cmake. Please be sure that you have enough memory free before building. My PC has 16GB of memory and it usually builds fine. In case it hangs during building, just wait for VSCode to throw an error and then restart the process.

To start building, usually, there should be a prompt to start this workspace as a container. Press OK. Otherwise, you can also do `ctrl + shift + p` then select `Dev Container: Rebuild and Reopen in Container` to start it manually.

The loading of the container will take some time. After the docker image has been downloaded, a script will be launch with automatically install the following packages:
- bullet_utils
- robot_properties_solo
- robot_properties_go2
- iterative_supervised_learning (with BiConMP)

The cmake build for iterative_supervised_learning will take a few minutes. Please be patient.

## Experiments
The experiments of this project are found in this folder [iterative_supervised_learning/examples/iterative_algorithm](iterative_supervised_learning/examples/iterative_algorithm/README.md)

## Files
|File|Usage|
|---|---|
|.devcontainer|devcontainer/docker specific config files|
|.vscode|setting file for TODO tree (optional)|
|bullet_utils|pybullet functions (taken from BiConMP)|
|iterative_supervised_learning|contains BiConMP and all the experiment scripts|
|robot_properties_go2|robot description and wrapper for unitree go2|
|robot_properties_solo|robot description and wrapper for solo12|

## Extras
After starting the devcontainer, it is also possible to run scripts from the terminal! To do that, make sure that you are in the project directory *quadrupedal_iterative_supervised_learning*, then you can run your scripts (XXX) as so:
```
devcontainer exec --workspace-folder . python3 iterative_supervised_learning/examples/iterative_algorithm/XXX
```

