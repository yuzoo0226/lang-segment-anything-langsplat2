import cv2
import numpy as np
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image

model = LangSAM()
image_pil = Image.open("./assets/car.jpeg").convert("RGB")
image_array = np.asarray(image_pil)

text_prompt = "wheel."
results = model.predict([image_pil], [text_prompt])

retults = results[0]

results_image = draw_image(image_rgb=image_array, masks=results["masks"], xyxy=results["boxes"], probs=results["scores"], labels=results["labels"])
results_image = cv2.cvtColor(results_image, cv2.COLOR_RGB2BGR)

cv2.imwrite("temp.png", results_image)
