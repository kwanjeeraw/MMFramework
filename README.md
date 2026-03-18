# MMFramework
An integrative framework for clinical metabolomics
MMFramework is a comprehensive platform for integrating metabolomics and clinical data that extends beyond conventional integration approaches to classification. 
It systematically compares clinical-only, metabolomics-only, and integrative models to quantify how each modality contributes to knowledge discovery. 
Through side-by-side evaluation, MMFramework identifies whether predictions are driven by physiological status, metabolic changes, or their combination. 
It provides an evidence-based foundation for selecting optimal modeling strategies, enabling more effective and informed use of multi-modal data for classification tasks.

Installation
=========
* Install [R software](https://www.r-project.org/)
* For DL models, install [Python](https://www.python.org/)
* Install the following packages in their respective environments

| **Required R packages** | **Required Python packages** |
| :--- | ---: |
| caret | argparse |
| circlize | dill |
| ComplexHeatmap | math |
| dplyr | matplotlib |
| forcats | numpy |
| ggplot2 | os |
| ggtext | pandas |
| metabox2 | random |
| optparse | seaborn |
| pROC | shap |
| qpdf | sklearn |
| RColorBrewer | sys |
| ropls | tensorflow |
| tibble |
| tidyr |
| tidytext |
| tidyverse | 

### Notes
* R packages can be installed via install.packages(), BiocManager::install, or remotes::install_github
* Python packages can be installed via pip or conda

Documentation
=========
See tutorial

License
=========
[GNU General Public License (v3)](https://github.com/kwanjeeraw/metabox2/blob/master/LICENSE)
