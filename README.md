# Vul2Image
To validate Vul2Image, our dataset
comprises 8,242 vulnerable and 14,676 non-vulnerable C/C++
functions from the SARD([Source code dataset link](https://pan.baidu.com/s/1_VFTR9FTq-pQOsy1SkQBRA?pwd=8878 )
). Moreover, we integrated datasets
from references [1], [2] and the National Vulnerability
Database (NVD) [3], totaling 935 real software functions
(160 vulnerable and 775 non-vulnerable). Accordingly, our
validation dataset consists of 8,402 vulnerable functions and
15,451 non-vulnerable functions and the corresponding dataset is linked as follows [Dataset](https://pan.baidu.com/s/16E3l0z2xAiZ8wq6tQskVBA?pwd=n3en). As shown in the following table.

![image](https://github.com/KLSEHB/Vul2image/assets/142284636/6706174a-5b1f-41e8-afd3-75e26f89e61f)

Moreover, we conducted vulnerability detection on multiple 
open-source systems and software, including Linux, FFmpeg,
and FreeType, spanning different versions of these systems and
software. Our analysis encompassed approximately 130,000
files, totaling around 67.35 million lines of the source code. In
the analysis, 134,750 files were converted into 1,731,072 RGB
images, and the corresponding dataset is linked as follows: [Open-source software dataset](https://pan.baidu.com/s/1_gMgkMhjs_xO1OiN_s8JVA?pwd=26ou). The software information we collect is as follows:

![image](https://github.com/KLSEHB/Vul2image/assets/142284636/2cba528c-930a-46ab-b22c-13fe058de843)

All corresponding RGB images produced are provided in this repository.

[1]Lin, Guanjun, Wei Xiao, Jun Zhang, and Yang Xiang. ”Deep learning-
based vulnerable function detection: A benchmark.” In Proceedings of
the 2019 International Conference on Information and Communications
Security (ICICS’19), pp. 219-232, 2019.

[2] Gkortzis, Apostolos, Dimitris Mitropoulos, and Diomidis Spinellis.
”VulinOSS: A Dataset of Security Vulnerabilities in Open-Source Sys-
tems.” In 2018 IEEE/ACM 15th International Conference on Mining
Software Repositories (MSR), 2018.

[3] National Vulnerability Database. https://nvd.nist.gov.
