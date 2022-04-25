# AAEO-MLP-GWL

## Cite our code

```code

@misc{nguyen_van_thieu_2022_6480834,
  author       = {Nguyen Van Thieu},
  title        = {AAEO-MLP-GWL},
  month        = {april},
  year         = {2022},
  publisher    = {Zenodo},
  version      = {v1.0.0},
  doi          = {10.5281/zenodo.6480834},
  url          = {https://doi.org/10.5281/zenodo.6480834}
}

```




## Data Statement:

1. The gridded rainfall and temperature data are available freely at 

```code 
https://www.imdpune.gov.in/Clim_Pred_LRF_New/Grided_Data_Download.html
```

2. The tidal height data are available at

```code 
https://www.psmsl.org/data/
```

3. The groundwater level data were provided on purchase by the the Department of Mines and Geology, Government of
   Karnataka state, India. So, the data is confidential and could not be shared publicly.


## Groundwater Level Modeling using Augmented Artificial Ecosystem Optimization

```code 
Compared Algorithms

1. Genetic Algorithm (GA)
2. Differential Evolution (DE)
3. Particle Swarm Optimization (PSO)
4. Harris Hawks Optimization (HHO)
5. Hunger Games Search (HGS)
6. Sparrow Search Algorithm (SSA)
7. Multi-Verse Optimizer (MVO)
8. Equilibrium Optimizer (EO)
9. Electromagnetic Field Optimization (EFO)

10. Forensic-Based Investigation Optimization (FBIO)
11. Coronavirus Herd Immunity Optimization (CHIO)

12. Slime Mould Algorithm (SMA)
13. Chaos Game Optimization (CGO)
14. Artificial Ecosystem-based Optimization (AEO)
15. Improved AEO
16. Modified AEO
17. Enhanced AEO
18. Adaptive AEO (Our proposed model)

```

## How to run models

```code 

1. All the models inside the script: script_mha_mlp.py
2. All the configuration inside the script: config.py
3. Calculate the statistics using the script: get_summary_statistics.py
4. For tradtional MLP run the script: script_mlp.py
5. Base class for all models is defined in models/based_mlp.py 
6. All the MHA-MLP model is defined in models/mha_mlp.py
7. All helper functions is located in utils
8. Results are located in data/input_data/results_paper
9. All figures are located in paper
10. The AAEO-MLP model figure used draw.io website to design

```



## Export environment

```code 
pip list --format=freeze > requirements.txt 

```

## Environment

```code 
conda create -n new ml python==3.7.5
conda activate ml
conda install -c conda-forge numpy
conda install -c conda-forge pandas
conda install -c conda-forge scikit-learn
conda install -c conda-forge matplotlib
conda install -c conda-forge tensorflow==2.1.0
conda install -c conda-forge keras==2.3.1

pip uninstall mealpy
pip uninstall permetrics
pip install mealpy==2.4.0
pip install permetrics==1.2.2
```



## Helper

https://pythontic.com/modules/pickle/dumps
https://medium.com/fintechexplained/how-to-save-trained-machine-learning-models-649c3ad1c018
https://github.com/keras-team/keras/issues/14180
https://ai-pool.com/d/how-to-get-the-weights-of-keras-model-

```python 
https://stackoverflow.com/questions/1894269/how-to-convert-string-representation-of-list-to-a-list

import json 
x = "[0.7587068025868327, 1000.0, 125.3177189672638, 150, 1.0, 4, 0.1, 10.0]"
solution = json.loads(x)
print(solution)

```


