### Introduction

This project provides Matlab code samples accompanying the paper "Online input design for discrimination of linear
models using concave minimisation" by Jacques Noom, Oleg Soloviev, Carlas Smith and Michel Verhaegen, submitted to International Journal of Control.



### Contents

The files containing "clafd" in the name are executable and produce for each true candidate model a .mat-file containing the generated simulation data. Files with "solve" in the name generate the file "OL_solution.mat" in the same folder. This file is already included in the parent directory for generating results for the open-loop approaches.



### Installation

The code was written in Matlab R2021a and Python 3.8 on Windows 10 with a 64-bit operating system and conda version 4.11.0. The scripts under the folder "polytopic_constraints" only rely on Matlab. (Most of) the scripts under the folder "quadratic_constraints" rely on both Matlab and Python. Therefore, please make sure that the release of Matlab is compatible with Python 3.8 (e.g. see https://www.mathworks.com/content/dam/mathworks/mathworks-dot-com/support/sysreq/files/python-compatibility.pdf (21 December 2023)) 



The Matlab scripts make use of the following toolboxes:
- Control System Toolbox
- Statistics and Machine Learning Toolbox



For running the code with quadratic constraints, first install the included environment "dccp38":

	conda env create -f environment.yml



The folder location of the environment "dccp38" can be found by entering in Anaconda Prompt:

	conda env list



Go to "path_to_environment\Lib\site-packages\dccp". Open the file "problem.py" and comment line 42 and 43:

    # if not is_dccp(self):
    #     raise Exception("Problem is not DCCP.")



Start a new Matlab session. Refer to the environment in the Matlab session using:

	pyenv('Version',"path_to_environment\python.exe")



Run the Matlab files.

### Citations

For citing the software and/or the methodology, please refer to the corresponding paper.
