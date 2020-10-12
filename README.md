# context-based-parking-slot-detect

Tensorflow implementation of [Context-based parking slot detection](https://ieeexplore.ieee.org/abstract/document/9199853) (IEEE Access)

This implementation is based on https://github.com/wizyoung/YOLOv3_TensorFlow


# Train Dataset
[link](https://drive.google.com/file/d/1i6I-71g1fNL7_Qh-Qs1oOKLclrP2qUmO/view?usp=sharing)

# Test Method

1. Download dataset
 - [link](https://drive.google.com/file/d/1z94Oqcy0Dich1GgiMkyPY5-wltsL8_hq/view?usp=sharing)
 - Unzip to $your_test_path
 
2. Download weight
 - [link](https://drive.google.com/file/d/1A6mdic0Rd8HgixM5CvJmW9VvRv1v8Ils/view?usp=sharing)
 - Unzip under main path (locate "weight_pcr" and "weight_psd" under "context-based detect" folder)
 
3. Prepare Dataset
 - python prepare_data.py --data_path $your_test_path

4. Evaluate
 - python --data_path $your_test_path
