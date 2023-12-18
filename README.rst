===========
snds_631_finalproject_DSilva
===========

Description
===========
This repo was developed for the final project of the class S&DS 631 at Yale University. It consists on reproducing the results from Singer and Yang, where they develop a framework for aligning 3D Density Maps in the context of Cryo-Electron Microscopy. 

Class-specific comments
===========

The code used to generate the plots in the report can be found in the folded report_results. To reproduce the results please download the map used (https://www.ebi.ac.uk/emdb/EMD-3683), and place it in a folder called "volumes/" inside "report_results". In addition, the results I obtained can be found in `Google Drive
<https://drive.google.com/file/d/1p1sXx0psoIxJsO0BGTx_c-Fa1k99J6AP/view?usp=drive_link/>`_. The results include the logs.

Requirements
===========
Required Python Libraries:

- Pymanopt
- Numpy
- Scipy
- Configargparse
- PyWavelets
- aspire

Installation
============
To install, first clone this repository
::

    git clone git@github.com:DSilva27/snds_631_finalproject_DSilva.git
    cd snds_631_finalproject

Then install using pip (use -e for editable mode)
::

    pip install -e .
