import torch
from transformers import GroundingDinoProcessor, GroundingDinoForObjectDetection, BitsAndBytesConfig
from torch.amp import autocast
from PIL import Image

from utils import pic2float, device, memo

# detector_id = "IDEA-Research/grounding-dino-tiny"

def preprocess_caption(caption: str) -> str:
    result = caption.lower().strip()
    if result.endswith("."):
        return result
    return result + "."


class GDINO:
    def __init__(self, model_id="IDEA-Research/grounding-dino-tiny", device="cuda"):
        self.device = device

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,                # включаем 4-битную квантизацию
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"         # можно попробовать и другие варианты, например 'fp4'
        )

        self.processor = GroundingDinoProcessor.from_pretrained(model_id)
        self.model = GroundingDinoForObjectDetection.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            quantization_config=quant_config,
            low_cpu_mem_usage=True,
        ).to(device)

    def detect_objects(self, image, text='a cat'):
        image = pic2float(image)
        text = preprocess_caption(text)
        inputs = self.processor(
            image,
            return_tensors="pt",
            text=text,
            do_rescale=False
        ).to(self.device)

        with torch.no_grad():
            with autocast('cuda', dtype=torch.float16):
                outputs = self.model(**inputs)

        results = self.processor.image_processor.post_process_object_detection(
            outputs,
            target_sizes=[image.shape[:-1]],
            threshold=0.1
        )[0]
        return results

print('Loading GDINO model...')
gdino = GDINO()
print('GDINO model loaded')

@memo
def detect_objects(image, text='a cat'):
    return gdino.detect_objects(image, text)

def test_gdino():
    test_image = "../image.jpg"
    test_image = Image.open(test_image).convert("RGB")
    print(detect_objects(test_image))

if __name__ == "__main__":
    test_gdino()