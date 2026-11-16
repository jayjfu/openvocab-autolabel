import torch
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


def build_prompt(class_names):
    return " . ".join(class_names) + " ."


def load_teacher(model_id):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
    return model, processor


@torch.inference_mode()
def detect(model, processor, image, prompt, box_threshold, text_threshold):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    result = processor.post_process_grounded_object_detection(
        outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold,
        target_sizes=[image.size[::-1]])[0]
    return result["boxes"].tolist(), result["scores"].tolist(), result["text_labels"]


def label_to_class(text_label, class_names):
    if text_label in class_names:
        return class_names.index(text_label)
    matches = [name for name in class_names if name in text_label]
    return class_names.index(max(matches, key=len)) if matches else None
