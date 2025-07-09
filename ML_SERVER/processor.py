# from ML_SERVER.sam import sam_process
from ML_SERVER.birefnet import biref_process
from ML_SERVER.blip import describe_center_object
from ML_SERVER.gdino import detect_objects

from utils import pic2pil, pic2float
from PIL import Image
from SHADOW.pix2pix import generate_shadow
from utils import memo, center, timer, decontaminate
from functools import lru_cache

@memo
def process_image(image, params):
    """
    Обработка изображения и текста.

    :param image: объект PIL.Image, изображение
    :param text: str, текст
    :return: tuple, объект PIL.Image и текст
    """
    image = pic2float(image)

    if 'rot' in params:
        rot = int(params['rot'])
    else:
        rot = None

    processed_images = remove_background_and_draw_shadow(image, rot)

    processed_images = [pic2pil(img) for img in processed_images]
    processed_text = params.get('text', '') + ' good'

    return processed_images, processed_text

def remove_background_and_draw_shadow(image, rot, max_objects=4):
    # blip
    t = timer()
    description = describe_center_object(image)
    timer(t,'Blip')

    # gdino
    t = timer()
    objects = detect_objects(image, description)
    timer(t, 'Gdino')
    bboxs = [bbox.tolist() for bbox in objects['boxes'][:max_objects]]


    processed_images = []

    # bi-ref
    t = timer()
    proc_images_and_masks = [biref_process(image, bbox) for bbox in bboxs]
    timer(t, 'BiRef')

    proc_images_and_masks = [center(*i) for i in proc_images_and_masks]

    t = timer()
    proc_images_and_masks = [(decontaminate(im,m), m ) for im, m in proc_images_and_masks]
    timer(t, 'Decontaminate')

    t = timer()
    for proc_image, mask in proc_images_and_masks:

        proc_image = generate_shadow(proc_image, mask, rots=rot)

        if isinstance(proc_image, list):
            processed_images.extend(proc_image)
            # processed_images.extend(mask)
        else:
            processed_images.append(proc_image)
            # processed_images.append(mask)
    timer(t, 'Shadow')
    return processed_images



def test():
    image = Image.open("..\image.jpg")
    params = {
        'text': 'test',
        'rot': '45'
    }
    processed_images, processed_text = process_image(image, params)
    print(processed_text)

if __name__ == '__main__':
    test()