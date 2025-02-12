from PIL import Image
from tqdm import tqdm
import os

TrainFilePath = '/home/liao/projects/codetranslate_linux/Pic_all/'
SaveFilePath = '/home/liao/projects/codetranslate_linux/Pic_150_all_23248/'

def main():

    file_list = os.listdir(TrainFilePath)

    for filename in tqdm(file_list, desc="Processing images"):
        original_image = Image.open(os.path.join(TrainFilePath, filename))

        width, height = original_image.size
        new_width, new_height = 150, 150

        new_image = Image.new('RGB', (new_width, new_height), (0, 0, 0))
        new_image.paste(original_image, (0, new_height - height))
        new_image.save(os.path.join(SaveFilePath, filename))

if __name__ =='__main__':
    main()