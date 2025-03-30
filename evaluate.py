import os
import cv2
import json
import numpy as np
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image, draw_mask
import argparse

class LangSamUtils():
    def __init__(self):
        self.model = LangSAM()

    def predict(self, path: str, prompt: str, return_mask=True) -> np.ndarray:
        """
        """
        image_pil = Image.open(path).convert("RGB")
        image_array = np.asarray(image_pil)

        text_prompt = prompt if prompt.endswith('.') else f"{prompt}."
        print(text_prompt)

        results = self.model.predict([image_pil], [text_prompt])
        results = results[0]

        if return_mask:
            results_image = draw_mask(image_rgb=image_array, masks=results["masks"], xyxy=results["boxes"], probs=results["scores"], labels=results["labels"])
        else:
            results_image = draw_image(image_rgb=image_array, masks=results["masks"], xyxy=results["boxes"], probs=results["scores"], labels=results["labels"])
            results_image = cv2.cvtColor(results_image, cv2.COLOR_RGB2BGR)
        
        return results_image

    def whole_predict(self, path: str, categories: str, output_dir=None) -> np.ndarray:
        """
        """
        image_pil = Image.open(path).convert("RGB")
        image_array = np.asarray(image_pil)

        text_prompt = ""
        for category in categories:
            category = category.replace(" ", "_")
            text_prompt += category if category.endswith('. ') else f"{category}. "

        if text_prompt.endswith(' '):
            text_prompt = text_prompt[:-1]
        
        print(text_prompt)

        results = self.model.predict([image_pil], [text_prompt])
        results = results[0]
        print(results["labels"])

        whole_results = {}

        for category in categories:
            check_category = category.replace(" ", " _ ").replace("-", " - ").lower()
            output_category = category.replace(" ", "_")
            for label in results["labels"]:
                print(label)
            filtered_masks = [mask for mask, label in zip(results["masks"], results["labels"]) if label == check_category]
            filtered_boxes = [box for box, label in zip(results["boxes"], results["labels"]) if label == check_category]
            filtered_probs = [prob for prob, label in zip(results["scores"], results["labels"]) if label == check_category]

            masked_image = draw_mask(image_rgb=image_array, masks=filtered_masks, xyxy=filtered_boxes, probs=filtered_probs, labels=[check_category])
            whole_results[output_category] = masked_image
            if output_dir is not None:
                output_path = os.path.join(output_dir, f"chosen_{output_category}.png")
                cv2.imwrite(output_path, masked_image)

        results_image = draw_image(image_rgb=image_array, masks=results["masks"], xyxy=results["boxes"], probs=results["scores"], labels=results["labels"])
        results_image = cv2.cvtColor(results_image, cv2.COLOR_RGB2BGR)

        output_path = os.path.join(output_dir, f"whole_masks.png")
        cv2.imwrite(output_path, results_image)

        return whole_results

    def run(self, args):
        """
        """

        prompts = args.prompts
        image_path = args.image_path
        labels_dir = args.labels_dir
        output_dir = args.output_dir

        if labels_dir is not None:
            if os.path.isdir(labels_dir):
                for file_name in os.listdir(labels_dir):
                    if file_name.endswith(".json"):
                        file_path = os.path.join(labels_dir, file_name)
                        with open(file_path, 'r') as json_file:
                            data = json.load(json_file)
                            image_path = os.path.join(labels_dir, data["info"]["name"])
                            name_without_extension = os.path.splitext(data["info"]["name"])[0]
                            current_output_dir = os.path.join(output_dir, name_without_extension)
                            os.makedirs(current_output_dir, exist_ok=True)
                            categories = [obj["category"] for obj in data["objects"]]
                            whole_results = self.whole_predict(image_path, categories, current_output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LangSAM with image and text prompts.")
    parser.add_argument("--image_path", "-i", type=str, default=None, help="Path to the input image.")
    parser.add_argument("--labels_dir", type=str, default=None, help="dir of labels.")
    parser.add_argument("--output_dir", type=str, default="outputs/", help="output_dir.")
    parser.add_argument("--prompts", type=str, nargs='+', default=None, help="List of text prompts.")
    args = parser.parse_args()

    cls = LangSamUtils()
    cls.run(args)
