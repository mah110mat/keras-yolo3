import os
import sys
from glob import glob
from PIL import Image

def resize_images(images_dir, image_save_dir, image_size):
    os.makedirs(image_save_dir, exist_ok=True)

    img_paths = glob(os.path.join(images_dir, '*'))

    for img_path in img_paths:
        # image open
        image = Image.open(img_path)
        rgb_im = image.convert('RGB')

        # resize image with unchanged aspect ratio using padding
        iw, ih = image.size
        w, h = (image_size,image_size)
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        # resize
        rgb_im = rgb_im.resize((nw,nh), Image.BICUBIC)

        # make background
        back_ground = Image.new("RGB", (image_size,image_size), color=(128,128,128))
        back_ground.paste(rgb_im, ((w-nw)//2, (h-nh)//2))

        # make path
        save_path = os.path.join(image_save_dir, os.path.basename(img_path))
        end_index = save_path.rfind('.')
        save_path = save_path[0:end_index]+'.png'
        print('save',save_path)
        back_ground.save(save_path,format='PNG')

def _main():
    images_dir = 'images/'  # input directory
    image_save_dir = 'resize_image/'  # output directory
    image_size = 416
    if len(sys.argv) > 1:
        image_size = int(sys.argv[1])

    resize_images(images_dir=images_dir, image_save_dir=image_save_dir, image_size=image_size)

if __name__ == '__main__':
    _main()
