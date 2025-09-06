# Nest-MGDL
This repository contains the natural image CS and sparse-view CT reconstruction pytorch codes for the following paper：  


### Environment  
```
pytorch <= 1.7.1 (recommend 1.7.0, 1.7.1)
scikit-image <= 0.16.2 (recommend 0.15.0, 0.16.2)
```

### 1.Test natural image CS    
1.1、Pre-trained models:  
All pre-trained models for our paper are in './model/'.  
1.2、Prepare test data:  
The original test sets are in './data/'.  
1.3、Prepare code:  
Open './Core-Nest-MGDL-natural-W-CS30-fullhis-p99-threesacle.py' and change the default run_mode to test in parser (parser.add_argument('--mode', type=str, default='test', help='train or test')).  
1.4、Run the test script (Core-Nest-MGDL-natural-W-CS30-fullhis-p99-threesacle.py).  
1.5、Check the results in './result/'.

### 2.Train natural image CS  
2.1、Prepare training data:  
We use the same datasets and training data pairs as ISTA-Net++ for CS. Due to upload file size limitation, we are unable to upload training data directly. Here we provide a [link](https://pan.baidu.com/s/1DY04Xsp7xfv2sJmm6DeTAA?pwd=y2l0) to download the datasets for you.  
2.2、Prepare measurement matrix:  
The measurement matrixs are in './sampling_matrix/'.  
2.3、Prepare code:  
Open './Core-Nest-MGDL-natural-W-CS30-fullhis-p99-threesacle.py' and change the default run_mode to train in parser (parser.add_argument('--mode', type=str, default='train', help='train or test')).  
2.4、Run the train script (Core-Nest-MGDL-natural-W-CS30-fullhis-p99-threesacle.py).  
2.5、Check the results in './log/'.


### Acknowledgements  
Thanks to the authors of ISTA-Net and ISTA-Net++, our codes are adapted from the open source codes of it.   

### Contact  
The code is provided to support reproducible research. If the code is giving syntax error in your particular python configuration or some files are missing then you may open an issue or directly email me at fanxiaohong@zjnu.edu.cn or fanxiaohong1992@gmail.com or fanxiaohong@smail.xtu.edu.cn
