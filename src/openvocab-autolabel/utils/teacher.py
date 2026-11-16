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



def build_class_prompt(processor, class_names, synonyms=None):
    phrases, phrase_class_ids = [], []
    for class_id, class_name in enumerate(class_names):
        for phrase in [class_name] + (synonyms or {}).get(class_name, []):
            phrases.append(phrase)
            phrase_class_ids.append(class_id)
    prompt = build_prompt(phrases)

    spans, position = [], 0
    for phrase in phrases:
        spans.append((position, position + len(phrase)))
        position += len(phrase) + len(" . ")
    offsets = processor.tokenizer(prompt, return_offsets_mapping=True)["offset_mapping"]
    phrase_tokens = [[token_index for token_index, (start, end) in enumerate(offsets)
                      if end > start and start >= span_start and end <= span_end]
                     for span_start, span_end in spans]
    return {"prompt": prompt, "phrase_tokens": phrase_tokens, "phrase_class_ids": phrase_class_ids,
            "num_classes": len(class_names)}


@torch.inference_mode()
def detect_classes(model, processor, image, class_prompt, min_score):
    inputs = processor(images=image, text=class_prompt["prompt"], return_tensors="pt").to(model.device)
    outputs = model(**inputs)
    token_probs = outputs.logits[0].sigmoid()
    class_scores = torch.zeros(token_probs.shape[0], class_prompt["num_classes"], device=token_probs.device)
    for tokens, class_id in zip(class_prompt["phrase_tokens"], class_prompt["phrase_class_ids"]):
        phrase_scores = token_probs[:, tokens].max(dim=1).values
        class_scores[:, class_id] = torch.maximum(class_scores[:, class_id], phrase_scores)
    scores, class_ids = class_scores.max(dim=1)
    keep = scores >= min_score

    width, height = image.size
    scale = torch.tensor([width, height, width, height], device=token_probs.device)
    center_x, center_y, box_width, box_height = (outputs.pred_boxes[0][keep] * scale).unbind(1)
    boxes = torch.stack([center_x - box_width / 2, center_y - box_height / 2,
                         center_x + box_width / 2, center_y + box_height / 2], dim=1)
    return boxes.tolist(), scores[keep].tolist(), class_ids[keep].tolist()