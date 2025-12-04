import io
import typing
import requests

import numpy as np
import torch


from PIL import Image
import cv2

from constant import device, MAX_MAMO_SIZE
import time

def memo(func):
    m = {}
    def wrapper(*arg, **kwarg):
        if len(m) > MAX_MAMO_SIZE:
            m.clear()
        hash = hash_all((*arg, *kwarg.items()))
        if hash not in m:
            m[hash] = func(*arg, **kwarg)
        return m[hash]

    return wrapper

def timer(start=None, text=None):
    # return start time in start is none
    # return length in sec if start is not None
    if start is None:
        return time.time()
    print(f'{text}: {time.time() - start:.2f} sec')

@memo
def decontaminate(im, mask, steps=15, blur=9):

    if isinstance(im, list):
        return [decontaminate(i, m, steps, blur) for i, m in zip(im, mask)]

    # приводим к float [0,1] и нужным формам
    im = pic2float(im)      # твоя функция, оставляем
    mask = pic2float(mask)

    # маска Photoroom ожидается однослойная (H, W) в [0,1]
    if mask.ndim == 3:
        alpha = mask[..., 0]
    else:
        alpha = mask

    alpha = np.clip(alpha, 0.0, 1.0)

    # Photoroom-реализация ждёт image в [0,1], shape (H,W,3)
    # im у тебя, судя по коду, уже такой. На всякий случай:
    if im.ndim == 2:
        im = np.stack([im, im, im], axis=-1)

    # маппим твой blur -> радиусы r1, r2
    # steps можно игнорить или использовать как множитель, но для интерфейса его оставим
    r1 = max(int(blur * 8), 3)
    if r1 % 2 == 0:
        r1 += 1
    r2 = max(int(blur * 2), 3)
    if r2 % 2 == 0:
        r2 += 1

    # напрямую вызываем numpy/cv2-реализацию из Photoroom (она у тебя уже есть)
    im_un = FB_blur_fusion_foreground_estimator_pil_2(im, alpha, r1=r1, r2=r2)

    # гарантируем тип
    im_un = im_un.astype(np.float32)

    return im_un

def FB_blur_fusion_foreground_estimator_pil_2(image, alpha, r1=90, r2=6):
    # Thanks to the source: https://github.com/Photoroom/fast-foreground-estimation
    alpha = alpha[:, :, None]
    F, blur_B = FB_blur_fusion_foreground_estimator_pil(
        image, image, image, alpha, r=r1)
    return FB_blur_fusion_foreground_estimator_pil(image, F, blur_B, alpha, r=r2)[0]


def FB_blur_fusion_foreground_estimator_pil(image, F, B, alpha, r=90):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    blurred_alpha = cv2.blur(alpha, (r, r))[:, :, None]

    blurred_FA = cv2.blur(F * alpha, (r, r))
    blurred_F = blurred_FA / (blurred_alpha + 1e-5)

    blurred_B1A = cv2.blur(B * (1 - alpha), (r, r))
    blurred_B = blurred_B1A / ((1 - blurred_alpha) + 1e-5)
    F = blurred_F + alpha * (image - alpha * blurred_F - (1 - alpha) * blurred_B)
    F = np.clip(F, 0, 1)
    return F, blurred_B

@memo
def decontaminate_old(im, mask, steps=15, blur=9):

    if isinstance(im, list):
        ims = []
        for i, m in zip(im, mask):
            ims.append(decontaminate(i, m, steps, blur))
        return ims

    im = pic2float(im)
    mask = pic2float(mask)

    if len(mask.shape) == 2:
        mask = np.stack([mask, mask, mask], axis=-1)

    hm1 = (mask > 0.999).astype(np.float32)

    im_un = im

    for _ in range(steps):
        im_un = mask_blur(im_un, hm1, blur)
    # im = im_un*hm0 + im*(1-hm0)

    return im_un


def ummult(im, m):
    # im = pic2float(im)
    # m = pic2float(m)
    e = 0.00001
    return im / (m + e)


def mask_blur(im, m, b=3):
    # black = np.zeros_like(im)
    im_m = im * m
    im_m = cv2.GaussianBlur(im_m, (b, b), 0)
    im_b = cv2.GaussianBlur(im, (b, b), 0)
    mb = cv2.GaussianBlur(m, (b, b), 0)
    im_un = ummult(im_m, mb)
    hmb0 = mb > 0.001
    im_b = hmb0 * im_un + (1 - hmb0) * im_b

    im = im * m + (1 - m) * im_b

    return im



def pic2int(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))

    if isinstance(image, Image.Image):
        image = np.array(image)

    pic_max = image.max()
    pic_min = image.min()

    if pic_min < 0 or pic_max > 255:
        e = 1e-8
        image = (image - pic_min + e) / (pic_max - pic_min + e)
        pic_max = image.max()

    if pic_max <= 1:
        image = image * 255

    return image.astype('uint8')


def pic2float(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))

    if isinstance(image, Image.Image):
        image = np.array(image)

    pic_max = image.max()
    pic_min = image.min()

    if pic_min < 0:
        e = 1e-8
        image = (image - pic_min + e) / (pic_max - pic_min + e)
    elif pic_max > 1:
        image = image / 255
    else:
        image = image.astype('float32')

    return image


def pic2pil(image):
    if isinstance(image, torch.Tensor):
        image = image.cpu().detach().numpy()
        image = image.transpose((1, 2, 0))
    if isinstance(image, np.ndarray):
        image = pic2int(image)
        image = Image.fromarray(image)
    return image

def pic2tensor(image):
    image = pic2float(image)
    image = torch.tensor(image).to(device).permute(2, 0, 1).float()
    return image


def swimg(image_arrays, server_url="http://127.0.0.1:9002/upload"):
    """
    Отправляет список numpy массивов на сервер Flask.

    :param image_arrays: Список numpy массивов (изображений)
    :param server_url: URL сервера Flask
    """

    image_arrays = [pic2int(i) for i in image_arrays]
    files = []

    for idx, array in enumerate(image_arrays):
        # Преобразуем numpy массив в изображение (если нужно, приводим к uint8)
        # if array.dtype != np.uint8:
        #    array = (array * 255).astype(np.uint8)

        # Если изображение черно-белое, добавляем канал
        if len(array.shape) == 2:  # Grayscale (H, W)
            array = np.stack([array] * 3, axis=-1)  # Convert to (H, W, 3)

        # Преобразуем массив в изображение с помощью PIL
        image = Image.fromarray(array)

        # Сохраняем изображение в буфер памяти
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        # Добавляем изображение в список файлов
        files.append(('images', (f'image_{idx}.png', buffer, 'image/png')))

    # Отправляем POST-запрос с изображениями
    response = requests.post(server_url, files=files)

    # Проверяем статус
    # if response.status_code == 200:
    #     print(f"Successfully sent {len(image_arrays)} images to the server!")
    # else:
    #     print(f"Failed to send images! Server response: {response.text}")


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def hash_image(image):
    image = pic2int(image)
    image = Image.fromarray(image)
    image = image.resize((512, 512))
    image = image.convert("RGB")
    image = image.tobytes()
    return hash(image)


def hash_all(x):
    if isinstance(x, typing.Union[list, tuple]):
        return hash(sum(hash_all(i) for i in x))

    if isinstance(x, dict):
        return hash(sum(hash_all(v) for v in x.values()))

    if isinstance(x, np.ndarray):
        return hash(x.tobytes())

    if isinstance(x, torch.Tensor):
        return hash(x.cpu().detach().numpy().tobytes())

    if isinstance(x, Image.Image):
        return hash_image(x)

    if isinstance(x, typing.Hashable):
        return hash(x)

    return hash(str(x))


def mask_crop(image, mask):
    coords = np.where(mask)
    y_min, y_max = coords[0].min(), coords[0].max()
    x_min, x_max = coords[1].min(), coords[1].max()
    return image[y_min:y_max, x_min:x_max], mask[y_min:y_max, x_min:x_max]


def center(image, mask, shape=(512, 512), boudary=0.2):
    h, w = shape
    bh = int(h * (1 - boudary))
    bw = int(w * (1 - boudary))
    obj_h, obj_w, _ = image.shape
    scale = min(bh / obj_h, bw / obj_w)
    new_h, new_w = int(obj_h * scale), int(obj_w * scale)
    # display(pic2pil(image))
    if scale < 1:
        algo = cv2.INTER_AREA
    else:
        algo = cv2.INTER_AREA
        algo = cv2.INTER_LINEAR
        # algo = cv2.INTER_CUBIC
        # algo = cv2.INTER_LANCZOS4

    image = cv2.resize(image, (new_w, new_h), interpolation=algo)

    mask = cv2.resize(mask, (new_w, new_h), interpolation=algo)

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    new_mask = np.zeros((h, w, 3), dtype=np.float32)
    new_mask[top:top + new_h, left:left + new_w] = mask
    #
    new_image = np.ones((h, w, 3), dtype=np.float32)
    new_image[top:top + new_h, left:left + new_w] = image

    # bg = np.ones((h, w, 3), dtype=np.float32)
    # new_image = new_image * new_mask + bg * (1 - new_mask)

    return new_image, new_mask