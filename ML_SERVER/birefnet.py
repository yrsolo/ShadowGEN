import numpy as np

import torch
from torchvision import transforms

from transformers import AutoModelForImageSegmentation, BitsAndBytesConfig

from PIL import Image
from utils import pic2float, pic2int, pic2pil, sigmoid, swimg, mask_crop, memo

MODEL_NAME = 'ZhengPeng7/BiRefNet'
DTYPE = torch.float16


class BiRef_Predictor():
    def __init__(self, model=None, processor=None, device=None, model_name=MODEL_NAME, type=DTYPE):
        self.dtype = type

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # включаем 4-битную квантизацию
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"  # можно попробовать и другие варианты, например 'fp4'
        )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if model is None:
            model = AutoModelForImageSegmentation.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=self.dtype,
                quantization_config=quant_config
            )

        self.model = model.to(self.device)
        self.processor = processor

        self.proc_size = (1024, 1024)
        self.processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.proc_size),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image, sale_mask=False):
        image = pic2float(image)
        org_size = image.shape[:2]
        input = self.processor(image).to(self.dtype).unsqueeze(0).to("cuda")

        with torch.inference_mode():
            mask = self.model(input)[-1].sigmoid()

        if sale_mask:
            mask = transforms.Resize(org_size)(mask)

        return mask.squeeze().cpu().numpy()

print('Loading BiRefNet model...')
biref_predictor = BiRef_Predictor()
print('BiRefNet model loaded!')

@memo
def biref_process(image, bbox=None):
    image = pic2float(image)
    if bbox:
        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = int(bbox[2])
        y2 = int(bbox[3])

        image = image[y1:y2, x1:x2]

    mask = biref_predictor.predict(image)
    mask = np.stack([mask]*3, axis=-1)
    return image, mask.astype(np.float32)

def test():
    test_image = "../image.jpg"
    test_image = Image.open(test_image).convert("RGB")
    test_bbox = [ 58.209,  29.224,  291.98,  183.45]
    print(biref_process(test_image, bbox=test_bbox))

if __name__ == "__main__":
    test()