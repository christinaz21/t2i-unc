class ClipScorer:
    def __init__(self, model_name: str = "openai/clip-vit-large-patch14"):
        # load model + preprocess here
        ...

    def image_text_similarity(self, image, text: str) -> float:
        ...

    def image_image_similarity(self, img1, img2) -> float:
        ...
