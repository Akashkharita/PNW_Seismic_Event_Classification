- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)



# **Seismic Event Classification in the Pacific Northwest**  

📌 **Created by**: @Akash Kharita (PhD Candidate, University of Washington)  

This repository provides a framework for **automated seismic event classification** in the Pacific Northwest. We train multiple **machine learning (ML) and deep learning (DL) models** on a dataset spanning **2001–2023**, containing **200K+ events** across four classes:  

1. **Earthquakes**  
2. **Explosions**  
3. **Noise**  
4. **Surface Events**  

### **📍 Geographical Distribution**  
![Seismic events in the Pacific Northwest](Figures/Figure_1.png)  

The primary objective is to **evaluate and compare various ML and DL approaches** to improve surface event classification while balancing accuracy, interpretability, and efficiency. The key differences between ML and DL techniques are illustrated below:  

![ML vs DL](Figures/ML_vs_DL.png)  

---

## **🚀 Installation**  

For cloud-based execution, refer to [HPSBook](https://seisscoped.org/HPS-book/chapters/cloud/AWS_101.html).  

### **1️⃣ Set up environment**  
```bash
sudo yum install -y git  
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
chmod +x Miniconda3-latest-Linux-x86_64.sh  
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda  
./miniconda/bin/conda init bash  
bash  
sudo yum groupinstall -y "Development Tools"
```


### **2️⃣ Clone the repository & install dependencies** 

```
git clone https://github.com/Akashkharita/PNW_Seismic_Event_Classification.git  
cd PNW_Seismic_Event_Classification  
conda create -y -n surface python=3.9.5  
conda activate surface  
pip install -r requirements.txt  
pip install jupyter  
conda install -y ipykernel  
python -m ipykernel install --user --name=surface  
```

### **3️⃣ Launch Jupyter Notebook**

- **Cloud**
  ```
  bash
  jupyter notebook --ip 0.0.0.0 --allow-root  
```

- **Local**
  ```
  bash
  jupyter notebook
  ```


Then we will clone the github repository and  install the required dependencies

```
git clone https://github.com/Akashkharita/PNW_Seismic_Event_Classification.git
cd PNW_Seismic_Event_Classification
conda create -y -n surface python=3.9.5
conda activate surface
pip install -r requirements.txt
pip install jupyter
conda install -y ipykernel
python -m ipykernel install --user --name=surface
```

Now we are all set to go! 😃


If you are on cloud run this - 

```
jupyter notebook --ip 0.0.0.0 --allow-root
```

If you are on local machine, just run this - 

```
jupyter notebook
```



## Usage

**Classic Machine Learning**
---

Classic machine learning involves the intermediate step of feature extraction. These features are extracted manually and are easier/more direct to interpret. Following is a description of various notebooks. 
 - The [notebook](notebooks/classification_based_on_physical_features_only.ipynb) shows classification of seismic events using **physics based** and **manual features**. It illustrates the process of downloading features from online repository, processing them, hyperparameter tuning and performance evaluation for balanced and unbalanced datasets at station and event level. It also has sections for feature importance and feature selection.
 - This [notebook](notebooks/classification_based_on_tsfel_features_only.ipynb) shows the same steps but for **tsfel** and **manual** features without including the sections for feature importance.
 - This [notebook](notebooks/classification_based_on_scatnet_features.ipynb) shows the same steps as above notebook but for **scatnet** and **manual** features.
 - This [notebook](notebooks/classification_based_on_combination_of_physical_tsfel_features.ipynb) shows the same steps as above notebook but for combination of **tsfel** and **physical** features
 - This [notebook](notebooks/comparison_of_ml_algorithms.ipynb) shows the comparison of performance of various machine learning algorithms in terms of accuracies, f1-scores and the computaitonal times for the combination of tsfel and physical features.
 - This [notebook](notebooks/testing_with_diff_freq_samp_duration.ipynb) shows the effects of different window lengths, sampling rate and frequencies of the input waveforms on the performance of ML classifier. 
 - The [notebook](notebooks/ML_Classification_Workflow_for_Scoped.ipynb) shows an example of how to process, train and tune the machine learning model and evaluate the results. I also show the importance of including manual parameters in improving the classification performance, in addition to this, I show that this process can also be used to identify potentially mislabeled events. 

** Feature Extraction Scripts**
We tested wide variety of feature sets commonly used in the seismological studies and their combinations. Each of these feature sets are different and extracted using different scripts from the waveforms that underwent same processing. Some scripts are divided into four parts, because of computational constraints. 
- The [script](feature_extraction_scripts/physical_feature_extraction_scripts/physical_feature_extraction_combined_script.py) and [script](feature_extraction_scripts/physical_feature_extraction_scripts/seis_feature.py) shows physical feature extraction. These are the features used and tested in several studies that involve Random Forests e.g. (Hibert et al. 2017 and the references therein)
- The [script](feature_extraction_scripts/tsfel_feature_extraction_scripts/tsfel_feature_extraction_combined_script.py) shows tsfel feature extraction. These involve extraction of over 300 statistical, temporal and frequency based features, only five of these features overlap with physical features
- The [script](feature_extraction_scripts/scatnet_feature_extraction_scripts/scatnet_feature_extraction_comcat_part1_p_50_100.py) shows scatnet feature extraction. These features involve application of scattering network and are very popular in unsupervised machine learning studies. 

**Deep Learning**
---
Deep learning involves automatic feature extraction which are harder to interpret but are also faster. 
- This [script](https://github.com/Akashkharita/PNW_Seismic_Event_Classification/blob/main/deep_learning/scripts/neural_network_architectures.py) contains all the neural network architectures
- This [notebook](deep_learning/testing_deep_learning_architectures.ipynb) contains training and validation process of all the architectures.
- This [notebook](deep_learning/testing_on_a_common_test_dataset.ipynb) contains the testing of all the neural network architectures on the common test dataset. 



## Contributing
Anyone is welcome to contribute to improve the codes and visualization of the results. I am available at my email  - ak287@uw.edu for further collaboration. 

## License

The repository has an MIT License. 
