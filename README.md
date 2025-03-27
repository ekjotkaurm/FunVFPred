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

# Data_splitting directory

This directory contains python code that split the dataset into train, test and validation datasets - split_aac.py

# Classifier directory

This directory contains the python codes to classify the dataset as virulent or non-virulent proteins.

(i) RF_aac.py - The Random Forest classifier is an ensemble learning algorithm that enhances accuracy and minimizes overfitting by combining predictions from multiple randomly constructed decision trees.

(ii) RF_5fold_cross_validation_aac.py - Random Forest with 5-fold cross-validation divides the dataset into five parts, using four for training and one for testing in each cycle to enhance reliability.

# FILES directory

This directory contains all the output files generated using the codes available in the Feature_extraction, Feature_fusion, Data_splitting and Classifier directories. The outut files are below mentioned.

(i) AAC.csv.gz - file containing the 20-dimensional amino acid composition (aac) features of positive and negtive dataset. 

(ii) DDE.csv.gz - file containing the 400-dimensional dipeptide deviation from expected mean (dde) features of positive and negtive dataset. 

(iii) UNIREP.csv.gz - file containing the 1900-dimensional unified representation (unirep) features of positive and negtive dataset. 

(iv) merged_AAC_DDE.csv.gz - file containing the combined 420-dimensional features including amino acid composition (aac) and dipeptide deviation from expected mean (dde) of positive and negtive dataset. 

(v) merged_AAC_DDE_UNIREP.csv.gz - file containing the combined 2320-dimensional features including amino acid composition (aac), dipeptide deviation from expected mean (dde) and unified representation (unirep)  of positive and negtive dataset. 

(vi) merged_AAC_UNIREP.csv.gz -  file containing the combined 1920-dimensional features including amino acid composition (aac) and unified representation (unirep) of positive and negtive dataset. 

(vii) merged_DDE_UNIREP.csv.gz -  file containing the combined 2300-dimensional features including dipeptide deviation from expected mean (dde) and unified representation (unirep) of positive and negtive dataset. 

(viii) test_AAC.csv.gz - positive and negative dataset having aac feature values to test the model.

(ix) train_AAC.csv.gz - positive and negative dataset having aac feature values to train the model.

(x) validation_AAC.csv.gz - positive and negative dataset having aac feature values to validate the model.

# How to run FunVFPred pipeline

Download "FunVFPred" on your system using command

$git clone http://github.com/ekjotkaurm/FunVFPred.git



