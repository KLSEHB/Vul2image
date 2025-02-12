# Vul2image: A Quick Image-inspired and CNN-based Vulnerability Detection system

## Dataset
Our dataset is derived from the Software Assurance Reference Dataset (SARD). In addition to SARD, we have also utilized datasets from VulCNN and Devign. Links to the relevant sources are provided below.

![image](https://github.com/user-attachments/assets/430d9548-c09d-479e-a085-271897ad7671)

Vul2image dataset:  "https://pan.baidu.com/s/1ziFx3AspuWU89zZX8ChBbw?pwd=hbx4"

VulCNN dataset : "https://github.com/CGCL-codes/VulCNN/tree/main/dataset"

Devign dataset : "https://drive.google.com/file/d/1x6hoF7G-tSYxg8AFybggypLZgMGDNHfF/edit"


Moreover, we conducted vulnerability detection on multiple 
open-source systems and software, including Linux, FFmpeg,
and FreeType, spanning different versions of these systems and
software. Our analysis encompassed approximately 130,000
files, totaling around 67.35 million lines of the source code. In
the analysis, 134,750 files were converted into 1,731,072 RGB
images, and the corresponding dataset is linked as follows: [Open-source software dataset](https://pan.baidu.com/s/1_gMgkMhjs_xO1OiN_s8JVA?pwd=26ou) or https://pan.baidu.com/s/13SwIcuMM8WGIecgoM2PTBA?pwd=an2s. The software information we collect is as follows:

![image](https://github.com/user-attachments/assets/98a75f38-67dc-4a21-9629-5fde78c0b93c)


All corresponding RGB images produced are provided in this repository.


### Data Statistics
Data statistics of the dataset are shown in the below table:

| Vul2image    | #Examples |
|--------------|-------------|
| Non-Vul      |  15,006        |
| Vul               |  8,242          |

| VulCNN       | #Examples |
|--------------|-------------|
| Non-Vul      |  21,057        |
| Vul               |  12,303        |

| Devign         | #Examples |
|--------------|-------------|
| Non-Vul      |  14,858        |
| Vul               |  12,460        |

## How to run
#### Convert source code to image
```shell
cd ~/Vul2image
python CodeScript_new.py /home/username/Vul2image/codetranslate_linux/bin/x64/Release/codetranslate_linux.out /home/username/Vul2image/codetranslate_linux/config/1.txt /home/username/Vul2image/data/SRAD_data/ /home/username/Vul2image/codetranslate_linux/config/12.txt /home/username/Vul2image/codetranslate_linux/config/19.txt /home/username/Vul2image/codetranslate_linux/config/PicNum.txt /home/username/Vul2image/codetranslate_linux/config/PicNum2.txt /home/username/Vul2image/codetranslate_linux/Release_result/fun_Vul/ /home/username/Vul2image/codetranslate_linux/Release_result/fun_No-Vul/ /home/username/Vul2image/codetranslate_linux/Release_result/NotAllConvertIR/ /home/username/Vul2image/codetranslate_linux/Release_result/PVCF_Vul/ /home/username/Vul2image/codetranslate_linux/Release_result/PVCF_No-Vul/ /home/username/Vul2image/codetranslate_linux/Release_result/RunTime/ 0
```

#### Train CNN
```shell
cd ~/Vul2image/scr/CNN
python CNN_pytorch.py 
```

[1]Lin, Guanjun, Wei Xiao, Jun Zhang, and Yang Xiang. ”Deep learning-
based vulnerable function detection: A benchmark.” In Proceedings of
the 2019 International Conference on Information and Communications
Security (ICICS’19), pp. 219-232, 2019.

[2] Gkortzis, Apostolos, Dimitris Mitropoulos, and Diomidis Spinellis.
”VulinOSS: A Dataset of Security Vulnerabilities in Open-Source Sys-
tems.” In 2018 IEEE/ACM 15th International Conference on Mining
Software Repositories (MSR), 2018.

[3] National Vulnerability Database. https://nvd.nist.gov.
