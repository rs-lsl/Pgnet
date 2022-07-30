# Pgnet

***There's a improved version compared with the publication in Tgrs with the modification in the deduction of the PDIN block: https://arxiv.org/abs/2201.11318***

Pytorch implement of 'Unmixing based PAN guided fusion network for hyperspectral imagery'

Please change the HSI dataset and SRF path and run the 'main.py' to train the model.

The JiaXing dataset used in this work could be available at "https://aistudio.baidu.com/aistudio/datasetdetail/124050"

We also release the code in paddle version if you use paddle framework in research. "https://aistudio.baidu.com/aistudio/projectdetail/3383847". 

Please consider cite this paper if you find the dataset and code useful. 

If you have any suggestions or questions, please dont hesitate to contact the author: 'cug_lsl@cug.edu.cn'.

To demonstrate the superiority of the proposed methods, we would like to take a fair comparison between different methods on JIAXING dataset. The training dataset is accout for 80% and the rest is for testing (see detail on "https://aistudio.baidu.com/aistudio/projectdetail/3383847").

metrics:    PSNR,    SSIM,   SAM,    ERGAS,   SCC,    Q,     RMSE
Pgnet:    [36.4348, 0.9156, 0.0632, 0.5868, 0.8875, 0.4684, 0.0151]
RHDN: (https://github.com/Jiahuiqu/RHDN)
