# FunVFPred: Predicting fungal virulence factors using a unified representation learning model



FunVFPred fills a critical gap in fungal pathogenicity research by being the first computational approach designed specifically to predict fungal VFs, based on Random Forest classifier. The model's predictive performance was significantly enhanced through the incorporation of UniRep embeddings, either individually or in combination with traditional sequence-based features like AAC and DDE. 


# FunVFPred Workflow

![Figure1_workflow_final](https://github.com/user-attachments/assets/f10fd894-d3ce-4e83-bb85-f5ed972cc5ac)


# Clone this GitLab Repository

    git clone http://github.com/ekjotkaurm/FunVFPred 

    cd FunVFPred

# Requirements
## Dependencies

    python 3.9.12

    tensorflow 2.17.0

    numpy 1.23.5

    pandas 2.3.0

    torch 2.7.0

    tape_proteins 0.5

    joblib 1.6.dev0 
    
## Install Python and Required Packages
#### Python & pip installation (Ubuntu/Debian)

  1. Install Python 3:

    sudo apt update
    sudo apt install python3

  2. Install pip (Python Package Manager):
  
    sudo apt install python3-pip

  3. Verify Installation:

    python3 --version

    pip3 --version

#### Install Python Dependencies

After setting up Python and pip, install the required packages:

    pip3 install torch
    
    pip3 install biopython
    
    pip3 install tape_proteins

    pip3 install pandas

    pip3 install numpy 

    pip3 install joblib


# Repository Contents
## DATA directory

It contains below mentioned data files.

**(i) input.fasta -** positive dataset having virulent protein sequences and negative dataset having non-virulent protein sequences.

**(ii) labels.csv -** file containing positive and negative protein ids with labels.
  
## Feature_extraction directory

This directory contains python codes to extract the features.

**(i) aac.py -** python code to extarct amino acid composition (aac) feature. 

**(ii) dde.py -** python code to extarct dipeptide deviation from expected mean (dde) feature.

**(iii) unirep.py -** python code to extarct unified representation (unirep) feature.

## Feature_fusion directory

This directory contains python codes to merge the features.

**(i) merge_aac_dde.py -** python code to merge amino acid composition (aac) and dipeptide deviation from expected mean (dde) features. 

**(ii) merge_aac_dde_unirep.py -** python code to merge amino acid composition (aac), dipeptide deviation from expected mean (dde) and unified representation (unirep) features.

**(iii) merge_aac_unirep.py -** python code to merge amino acid composition (aac) and unified representation (unirep) features.

**(iv) merge_dde_unirep.py -** python code to merge dipeptide deviation from expected mean (dde) and unified representation (unirep) features.

## Data_splitting directory

This directory contains python code that split the dataset into train, test and validation datasets **- split_aac.py**

## Classifier directory

This directory contains the python codes to classify the dataset as virulent or non-virulent proteins.

**(i) RF_aac.py -** The Random Forest classifier is an ensemble learning algorithm that enhances accuracy and minimizes overfitting by combining predictions from multiple randomly constructed decision trees.

**(ii) RF_5fold_cross_validation_aac.py -** Random Forest with 5-fold cross-validation divides the dataset into five parts, using four for training and one for testing in each cycle to enhance reliability.

**(iii) ANN_aac.py -** Implements an Artificial Neural Network designed to classify fungal proteins using AAC features. The architecture includes dense layers with ReLU activation and dropout to reduce overfitting, ending with a sigmoid layer for binary classification.

**(iv) MLP_aac.py -** Builds a Multi-Layer Perceptron model using protein sequence features. The network includes multiple dense layers with ReLU activations, optimized for distinguishing between virulent and non-virulent proteins.

**(v) DNN_aac.py -**  Implements a Deep Neural Network using AAC features. The model architecture includes multiple hidden layers with dropout and activation functions to enhance learning and reduce overfitting during classification.


## FILES directory

This directory contains all the output files generated using the codes available in the Feature_extraction, Feature_fusion, Data_splitting and Classifier directories. The outut files are below mentioned.

**(i) AAC.csv.gz -** file containing the 20-dimensional amino acid composition (aac) features of positive and negtive dataset. 

**(ii) DDE.csv.gz -** file containing the 400-dimensional dipeptide deviation from expected mean (dde) features of positive and negtive dataset. 

**(iii) UNIREP.csv.gz -** file containing the 1900-dimensional unified representation (unirep) features of positive and negtive dataset. 

**(iv) merged_AAC_DDE.csv.gz -** file containing the combined 420-dimensional features including amino acid composition (aac) and dipeptide deviation from expected mean (dde) of positive and negtive dataset. 

**(v) merged_AAC_DDE_UNIREP.csv.gz -** file containing the combined 2320-dimensional features including amino acid composition (aac), dipeptide deviation from expected mean (dde) and unified representation (unirep)  of positive and negtive dataset. 

**(vi) merged_AAC_UNIREP.csv.gz -**  file containing the combined 1920-dimensional features including amino acid composition (aac) and unified representation (unirep) of positive and negtive dataset. 

**(vii) merged_DDE_UNIREP.csv.gz -**  file containing the combined 2300-dimensional features including dipeptide deviation from expected mean (dde) and unified representation (unirep) of positive and negtive dataset. 

**(viii) test_AAC.csv.gz -** positive and negative dataset having aac feature values to test the model.

**(ix) train_AAC.csv.gz -** positive and negative dataset having aac feature values to train the model.

**(x) validation_AAC.csv.gz -** positive and negative dataset having aac feature values to validate the model.

## Step 1: Preparing input files

**(i) input.fasta -** containg the positive (virulent protein sequences) and negative (non-virulent protein sequences) dataset.
Virulent proteins of fungal species can be downloaded from the virulence factors databases such as PHI-base (http://www.phi-base.org/), Victors (http://www.phidias.us/victors) and DFVF (http://sysbio.unl.edu/DFVF/) and non-virulent proteins of same fungal species from UniProt (https://www.uniprot.org/) database.
After downloading the dataset, remove the redundant protein sequences using CD-HIT with 100% identity threshold. 
Commands to prepare the input.

    cd-hit -i positive_proteins.fasta -o CDHIT100_pos -c 1 -T 8 -M 2000
    cd-hit -i negative_proteins.fasta -o CDHIT100_neg -c 1 -T 8 -M 2000
    
(ii) Next step involves balancing the imbalanced dataset. In real-life scenarios, virulent (positive) and non-virulent (negative) proteins are highly imbalanced, with more negative dataset, making data imbalance a challenge. To address this, we applied random undersampling to achieve a 1:1 ratio for balanced model training. The python code for balancing the dataset is mentioned below.

    python balance_data.py

    cat balanced_pos.fasta balanced_neg.fasta >input.fasta

**(iii) labels.csv -** labeled file in csv format containing balanced dataset of positive and negative protein ids.

    grep ">positive" input.fasta |sed "s/>//g"|sed "s/$/\,1/g" > positive.csv
    
    grep ">negative" input.fasta |sed "s/>//g"|sed "s/$/\,0/g" > negative.csv
    
    (echo "protein_ids,label"; awk -F',' 'NR>1 {print $1",1"}' positive.csv; awk -F',' 'NR>1 {print $1",0"}' negative.csv) > labels.csv
    
# How to run FunVFPred pipeline

   ## Step 2: Download "FunVFPred" on your system using command

    git clone http://github.com/ekjotkaurm/FunVFPred.git

   ### Change directory to the cloned FunVFPred folder 
    
    cd FunVFPred  

   ### Navigate into the FILES directory where input datasets are stored 
    
    cd FILES

   ### Decompress all .gz files in the directory
   
    gunzip *.gz
   
   ## Step 3: Run python code to extract features 

    cd Feature_extraction
    
    python aac.py

    python dde.py

    python unirep.py

   ## Step 4: Merge the extracted features 
     
    cd Feature_fusion
    
    python merge_aac_dde.py

    python merge_aac_dde_unirep.py

    python merge_aac_unirep.py

    python merge_dde_unirep.py

   ## Step 5: Splitting the dataset

Run the following command to split the dataset into training, testing, and validation sets:

    cd Data_splitting
    
    python split_aac.py

This script is specifically for the amino acid composition (AAC) feature file. For other feature types, such as DDE, UniRep, or merged features, use the same script but modify the input file accordingly.
   
   ## Step 6: Model Building/ Classification 

Run the following command to build the model to classify the virulent and non-virulent proteins: 


    cd Classifier
    
    python RF_aac.py

    python ANN_aac.py

    python DNN_aac.py

    python MLP_aac.py

This script is specifically for the amino acid composition (AAC) feature file. For other feature types, such as DDE, UniRep, or merged features, use the same script but modify the input file accordingly.

   ## Step 7: Five-fold cross-validation

Run the following command for 5-fold cross-validation:

    RF_5fold_cross_validation_aac.py

This script is designed for the amino acid composition (AAC) feature file. For other feature types, such as DDE, UniRep, or merged features, the same script can be used by simply changing the input file.


# Prediction of virulent and non-virulent proteins using unknown dataset
    
    cd FunVFPred
    
    python predict_virulence.py

