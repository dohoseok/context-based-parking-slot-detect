# context-based-parking-slot-detect

Tensorflow implementation of [Context-based parking slot detection](https://ieeexplore.ieee.org/abstract/document/9199853) (IEEE Access)

This implementation is based on https://github.com/wizyoung/YOLOv3_TensorFlow


# Prepare Dataset (PIL-park)
0. This code should be run only once at the beginning.

1. Download Train Dataset
 - [link](https://drive.google.com/file/d/1i6I-71g1fNL7_Qh-Qs1oOKLclrP2qUmO/view?usp=sharing)
 - Unzip to $your_data_path/train folder

2. Download Test Dataset
 - [link](https://drive.google.com/file/d/1z94Oqcy0Dich1GgiMkyPY5-wltsL8_hq/view?usp=sharing)
 - Unzip to $your_data_path/test folder
 
3. Data augmentation, create tfrecord and text files
 - python prepare_data.py --data_path=$your_data_path


# Train Dataset
1. Download pretrain weight (Updated 2020.10.26)
 - [link](https://drive.google.com/file/d/1g3PXkTn8-pmIotjJqX_aR1ZPJNrrmWKG/view?usp=sharing)
 - Save to 'pre_weight' folder under "context-based detect" folder
 
2. python train.py --data_path=$your_data_path

3. Trained Weight path
- Weight files of parking context recognizer are saved to 'weight_pcr/YYYYMMDD_HHMM'
- Weight files of parking slot detector fine-tuned for parallel parking slots are saved to 'weight_psd/type_0/YYYYMMDD_HHMM'
- Weight files of parking slot detector fine-tuned for perpendicular parking slots are saved to 'weight_psd/type_1/YYYYMMDD_HHMM'
- Weight files of parking slot detector fine-tuned for diagonal parking slots are saved to 'weight_psd/type_2/YYYYMMDD_HHMM'


# Test Method (with downloaded weight files)
1. Download trained weight
 - [link](https://drive.google.com/file/d/1A6mdic0Rd8HgixM5CvJmW9VvRv1v8Ils/view?usp=sharing)
 - Unzip under main path (locate "weight_pcr" and "weight_psd" under "context-based detect" folder)
 
2. Evaluate
 - python test.py --data_path=$your_test_path
 

# Test Method (with your trained weight files)
1. Evaluate
 - python test.py --data_path=$your_test_path --pcr_test_weight='weight_pcr/YYYYMMDD_HHMM' --psd_test_weight_type0='weight_psd/type_0/YYYYMMDD_HHMM' --psd_test_weight_type1='weight_psd/type_1/YYYYMMDD_HHMM' --psd_test_weight_type2='weight_psd/type_2/YYYYMMDD_HHMM'
 
