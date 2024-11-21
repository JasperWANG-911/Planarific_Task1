from transformers import CLIPProcessor, CLIPModel, Owlv2Processor, Owlv2ForObjectDetection, AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 1. CLIP
def detect_conservatory_using_CLIP(image_path, model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32"), processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")):

    # Load the image
    image = Image.open(image_path)

    # Scripts for identifying the conservatory(to be improved)
    inputs = processor(
        text=[
            "a glass top attached to a building",
            "a glass conservatory attached to a house, transparent roof and walls",
            "a greenhouse-like structure with large glass panels",
            "a sunroom with glass or reflective windows",
            "a garden structure with transparent or semi-transparent surfaces",
            "a building with a red tile roof",
            "a grass area"
        ],
        images=image,
        return_tensors="pt",
        padding=True
    )

    # Inference
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)

    # Probability of conservatory exisiting under different prompts
    prompt_probs = [probs[0, i].item() for i in range(6)]

    # threshold probbaility(to be improved)
    threshold = 0.1

    if all(prob > threshold for prob in prompt_probs[:5]):
        return "The image likely contains a conservatory."

    else:
        return "The image likely does not contain a conservatory."

# 2. OWLv2
def detect_conservatory_using_OWLv2(
    image_path,
    model=Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble"),
    processor=Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble"),
    threshold=0.1
):

    # Load the image
    image = Image.open(image_path)

    # Scripts
    texts = [
    "an outdoor extension with a glass structure attached to a house",
    "a small structure made of glass, typically attached to the rear of a house",
    "an enclosed garden room with glass walls and ceiling, adjacent to the main building",
    "an extension with glass windows, visible in an aerial view of a house"
    ]

    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True, truncation=True)

    # Perform model inference
    with torch.no_grad():
        outputs = model(**inputs)

    # Set target image size (height, width)
    target_sizes = torch.tensor([[image.size[1], image.size[0]]])
    results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=threshold)

    # Extract results for the first image
    i = 0
    boxes, scores, labels = results[i]["boxes"], results[i]["scores"], results[i]["labels"]

    # Plot the image
    fig, ax = plt.subplots(1, figsize=(12, 8))
    ax.imshow(image)

    # Draw detection boxes on the image
    for box, score, label in zip(boxes, scores, labels):
        if score > threshold:
            xmin, ymin, xmax, ymax = box.tolist()
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                     linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
            ax.text(xmin, ymin, f"{texts[label]}: {score:.2f}", color='red', fontsize=12, verticalalignment='top')

    plt.axis('off')
    plt.show()

# 3. Florence2
def detect_conservatory_using_Florence2(
        image_path,
        phrase,
        model_name="microsoft/Florence-2-base"):
    # Check if CUDA is available and set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model and processor
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    # Load and preprocess the image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Caption task prompt
    task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"

    # Prepare inputs
    inputs = processor(text=task_prompt + phrase, images=image, return_tensors="pt").to(device)

    # Generate outputs
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        do_sample=False,
        num_beams=3,
    )

    # Decode and post-process results
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    result = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    # Extract bounding boxes and draw them on the image
    bboxes = result['<CAPTION_TO_PHRASE_GROUNDING>']['bboxes']
    draw = ImageDraw.Draw(image)
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1 - 10), phrase, fill="red")

    # Display the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()