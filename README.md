# LAVQ-Editor
## Code for "Personalized Heart Disease Detection via ECG Digital Twin Generation" 

## 🚀 Installation 

You can install the remaining dependencies for our package by executing:
```
pip install -r requirements.txt
```
Please note, our package has been tested and confirmed to work with Python 3.7. We recommend using this version to ensure compatibility and optimal performance.

## Dataset
You can download our reprepared PTB-XL Dataset in [link]().
```
PublicDataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── reprepared
│   ├── Patient_Select_145_sclc_X_half1.csv
│   ├── Patient_Select_145_sclc_X_half1.npy
│   ├── Patient_Select_145_sclc_X_half2.csv
│   ├── Patient_Select_145_sclc_X_half2.npy
│   ├── Patient_Selected_291.csv
│   ├── Patient_Selected_291_sclc.csv
│   ├── Patient_Selected_291_sclc_X.npy
│   ├── Train_sclc.csv
│   ├── Train_sclc.npy
│   ├── Train_sclc_X.csv
│   ├── Train_sclc_X.npy
│   └── weight_5_scls.json
```


## 💻 Usage (3 lines of code)
To train LAVQ-Editor, you can use the following code.
```
python train.py --model_name LAVQ_Editor --batch_size 128 
```

## 💼 Support
If you need help with the tool, you can raise an issue on our GitHub issue tracker. For other questions, please contact our team. 



