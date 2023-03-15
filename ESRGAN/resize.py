from PIL import Image
from tqdm import tqdm
import glob

filename_list = glob.glob('.\\LR\\*.bmp')
filename_list.sort()

fill_number = len(str(len(filename_list)))
for idx, filename in enumerate(tqdm(filename_list), 1):
    im = Image.open(filename)

    
    area = (108, 92, 2034, 1438)
    crop_image = im.crop(area)
    savename = '.\\crop\\crop_0' + str(idx).zfill(fill_number) + '.jpg'
    crop_image.save(savename)