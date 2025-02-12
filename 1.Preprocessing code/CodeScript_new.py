from doctest import OutputChecker
import os
import sys
import time
import subprocess

TargetExePath = ''
InterMediateFilePath = ''
SourceFilePath = ''
OutPutFilePath3 = ''
OutPutFilePath4 = ''
OutputFilePath13 = ''
OutputFilePath14 = ''
OutputFilePath15 = ''
OutputFilePath18 = ''
OutPutPicNumConfigFilePath = ''
OutPutPicNumConfigFilePath2 = ''
PvcfFile = ''
NoPvcfFile = ''
RunTime = ''



if __name__ == '__main__':
    print('')
    if len(sys.argv) != 15:
        print('Command Fail')
        sys.exit(0)
    TargetExePath = sys.argv[1]
    InterMediateFilePath = sys.argv[2]
    SourceFilePath = sys.argv[3]
    OutPutFilePath3 = sys.argv[4]
    OutPutFilePath4 = sys.argv[5]
    OutPutPicNumConfigFilePath = sys.argv[6]
    OutPutPicNumConfigFilePath2 = sys.argv[7]
    OutPutFilePath13 = sys.argv[8]
    OutPutFilePath14 = sys.argv[9]
    OutPutFilePath18 = sys.argv[10]
    PvcfFile = sys.argv[11]
    NoPvcfFile = sys.argv[12]
    RunTime = sys.argv[13]
    Mod = sys.argv[14]

    timeout_seconds = 1
    # 下面的是针对old_file的遍历方式
    for source_file in os.listdir(SourceFilePath):
        SourceFile = os.path.join(SourceFilePath, source_file)
        print("\nSourceFile is: ", SourceFile, "\n")

        cmd = [
            TargetExePath,
            InterMediateFilePath,
            SourceFile,
            OutPutFilePath3,
            OutPutFilePath4,
            OutPutPicNumConfigFilePath,
            OutPutPicNumConfigFilePath2,
            OutPutFilePath13,
            OutPutFilePath14,
            OutPutFilePath18,
            PvcfFile,
            NoPvcfFile,
            RunTime,
            Mod
        ]
        try:
            # 使用subprocess.run()来执行命令，并设置超时
            result = subprocess.run(cmd, timeout=timeout_seconds, check=True)
            # 如果需要捕捉输出，可以添加 stdout=subprocess.PIPE 和 stderr=subprocess.PIPE 参数
        except subprocess.TimeoutExpired:
            print(f"Processing of {source_file} timed out after {timeout_seconds} seconds.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while processing {source_file}: {e}")


        # STR = TargetExePath + ' ' + InterMediateFilePath + ' ' + SourceFile + ' ' + OutPutFilePath3 + ' ' + OutPutFilePath4 + ' ' + OutPutPicNumConfigFilePath + ' ' + OutPutPicNumConfigFilePath2 + ' ' + OutPutFilePath13 + ' ' + OutPutFilePath14 + ' ' + OutPutFilePath18 + ' ' + PvcfFile + ' ' + NoPvcfFile + ' ' + RunTime + ' ' + Mod
        # #print("\n",STR,"\n")
        # os.system(STR)


    # 下面的是针对新数据集的遍历方式
    # files = os.listdir(SourceFilePath)
    # for file in files:
    #     file_path = os.path.join(SourceFilePath, file)#/home/liao/projects/codetranslate_linux/dataset/CWE190_Integer_Overflow
    #     print(file_path,"\n")
    #     for source_file in os.listdir(file_path):
    #         SourceFile = os.path.join(file_path, source_file)
    #         print("\nSourceFile is: ",SourceFile,"\n")
    #         #time.sleep(1)
    #         #Run2(SourceFilePath, OutPutFilePath3)
    #         STR = TargetExePath + ' ' + InterMediateFilePath + ' ' + SourceFile + ' ' + OutPutFilePath3 + ' ' + OutPutFilePath4 + ' ' + OutPutPicNumConfigFilePath + ' ' + OutPutPicNumConfigFilePath2 + ' ' + OutPutFilePath13 + ' ' + OutPutFilePath14 + ' ' + OutPutFilePath18 + ' ' + PvcfFile + ' ' + NoPvcfFile + ' ' + RunTime
    #         #print("\n",STR,"\n")
    #         os.system(STR)


    