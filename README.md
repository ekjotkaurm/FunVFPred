# FunVFPred: Predicting fungal virulence factors using a unified representation learning model



FunVFPred fills a critical gap in fungal pathogenicity research by being the first computational approach designed specifically to predict fungal VFs, based on Random Forest classifier. The model's predictive performance was significantly enhanced through the incorporation of UniRep embeddings, either individually or in combination with traditional sequence-based features like AAC and DDE. 


# FunVFPred Workflow

![Figure1_workflow_final](https://github.com/user-attachments/assets/f10fd894-d3ce-4e83-bb85-f5ed972cc5ac)


# Clone this GitLab Repository

    git clone http://github.com/ekjotkaurm/FunVFPred 

    cd FunVFPred

# Requirements
# Dependencies

    python 3.9.12

    tensorflow 2.17.0

    numpy 1.23.5
# Install Dependencies

Python:
  
  For Ubuntu/Debian-based Systems:
  
  1. Install Python 3:

    sudo apt install python3

  2. Install pip (Python Package Manager):
  
    sudo apt install python3-pip

  3. Verify Installation:

    python3 --version

    pip3 --version
  
# Repository Contents
# DATA directory

It contains below mentioned data files.

(i) input.fasta - positive dataset having virulent protein sequences and negative dataset having non-virulent protein sequences.

(ii) labels.csv - file containing positive and negative protein ids with labels.
  
# Feature_extraction directory

This directory contains python codes to extract the features.

(i) aac.py - python code to extarct amino acid composition (aac) feature. 

(ii) dde.py - python code to extarct dipeptide deviation from expected mean (dde) feature.

(iii) unirep.py - python code to extarct unified representation (unirep) feature.

# Feature_fusion directory

This directory contains python codes to merge the features.

(i) merge_aac_dde.py - python code to merge amino acid composition (aac) and dipeptide deviation from expected mean (dde) features. 

(ii) merge_aac_dde_unirep.py - python code to merge amino acid composition (aac), dipeptide deviation from expected mean (dde) and unified representation (unirep) features.

(iii) merge_aac_unirep.py - python code to merge amino acid composition (aac) and unified representation (unirep) features.

(iv) merge_dde_unirep.py - python code to merge dipeptide deviation from expected mean (dde) and unified representation (unirep) features.
