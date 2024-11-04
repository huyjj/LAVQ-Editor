# LAVQ-Editor
## Code for IJCAI2024 paper ["Personalized Heart Disease Detection via ECG Digital Twin Generation"](https://www.ijcai.org/proceedings/2024/649)

## 🚀 Installation 

You can install the remaining dependencies for our package by executing:
```
pip install -r requirements.txt
```
Please note, our package has been tested and confirmed to work with Python 3.7. We recommend using this version to ensure compatibility and optimal performance.

## 📂 Dataset
You can download our reprepared PTB-XL Dataset in [BaiduDisk](https://pan.baidu.com/s/1WrthCxT_UKm5iJacYtcMDQ?pwd=n2id) or [Huggingface](https://huggingface.co/datasets/Yaojunhu/LAVQEditor-PTBXL/). 
```
ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── reprepared
│   ├── Patient_Select_145_sclc_X_half1.csv
│   ├── Patient_Select_145_sclc_X_half1.npy
│   ├── Patient_Select_145_sclc_X_half1_crop_wave.npy
│   ├── Patient_Select_145_sclc_X_half2.csv
│   ├── Patient_Select_145_sclc_X_half2.npy
│   ├── Patient_Select_145_sclc_X_half2_crop_wave.npy
│   ├── Patient_Selected_291.csv
│   ├── Patient_Selected_291_sclc.csv
│   ├── Patient_Selected_291_sclc_X.npy
│   ├── Train_sclc_X.csv
│   ├── Train_sclc_X.npy
│   ├── Train_sclc_X_crop_wave.npy
│   └── weight_5_scls.json
```


## 💻 Usage
Before starting the training, a classifier is needed to monitor the training process.
However, the quality of this classifier is not important.
You can choose anyway to obtain the best_w.pth.
To train LAVQ-Editor, you can use the following code.
```
python train.py --model_name LAVQ_Editor --batch_size 128 
```

## 💼 Support
If you need help with the tool, you can raise an issue on our GitHub issue tracker. For other questions, please contact our team.

## 📜 Citation

If you find our project useful, please cite the following paper:

```bibtex
@inproceedings{ijcai2024p649,
  title     = {Personalized Heart Disease Detection via ECG Digital Twin Generation},
  author    = {Hu, Yaojun and Chen, Jintai and Hu, Lianting and Li, Dantong and Yan, Jiahuan and Ying, Haochao and Liang, Huiying and Wu, Jian},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {5872--5881},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/649},
  url       = {https://doi.org/10.24963/ijcai.2024/649},
}