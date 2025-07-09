import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, BitsAndBytesConfig

from utils import pic2float, device, memo

BLIP = "Salesforce/blip2-opt-2.7b"
BLIP_PROMPT = "Question: Describe the main (central) object in this image. Answer:"
# 'Focus on the single most prominent object in the center of the photo. What is it?'


class BlipImageCaptioner:
    def __init__(self,
                 device: str = device,
                 model_name: str = BLIP,
                 processor_name: str = BLIP,
                 prompt: str = BLIP_PROMPT,
                 # prompt: str =
                 ):
        self.device = device

        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,  # включаем 4-битную квантизацию
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_quant_type="nf4"  # можно попробовать и другие варианты, например 'fp4'
        )

        self.processor = Blip2Processor.from_pretrained(
            model_name,
        )
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            processor_name,
            torch_dtype=torch.float16,
            quantization_config=quant_config
        ).to(device)
        self.prompt = prompt

    def question(self, image) -> str:
        image = pic2float(image)
        prompt = self.prompt
        inputs = self.processor(image, prompt, return_tensors="pt", do_rescale=False).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )

        caption = self.processor.decode(out[0], skip_special_tokens=True)
        answer_idx = caption.find("Answer:") + len("Answer: ")
        return caption[answer_idx:]

print('Loading Blip model...')
blip = BlipImageCaptioner()
print('Blip model loaded')

@memo
def describe_center_object(image):
    return blip.question(image)

def test_blip():
    test_image = "../image.jpg"
    test_image = Image.open(test_image).convert("RGB")
    print(describe_center_object(test_image))

if __name__ == "__main__":
    test_blip()