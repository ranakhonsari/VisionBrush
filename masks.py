from PIL import Image
import numpy as np
import torch
from lang_sam import LangSAM

class MaskGenerator:
    def __init__(self):
        self.model = LangSAM()

    def segmentation_model(self, image_path, mask_text_prompt):
        """Generate masks for the given image and text prompt."""
        # Load the image
        image_pil = Image.open(image_path).convert("RGB")

        # Get predictions
        results = self.model.predict([image_pil], [mask_text_prompt])

        # Extract masks from the results
        masks = results[0]['masks']  # Predicted masks (binary masks)
        return masks, image_pil

    @staticmethod
    def inpainting_masks(masks, image_pil):
        """Combine masks and prepare them for inpainting."""
        # Combine all masks into a single mask
        combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
        for mask in masks:
            combined_mask = np.logical_or(combined_mask, mask)  # Combine masks

        # Convert the combined mask to a PIL image
        mask_image = Image.fromarray((combined_mask * 255).astype(np.uint8))

        # Resize the original image and mask to 512x512
        init_image = image_pil.resize((512, 512))
        mask_image = mask_image.resize((512, 512))
        return init_image, mask_image

    @staticmethod
    def make_inpaint_condition(init_image, mask_image):
        """Prepare the control image for inpainting."""
        init_image = np.array(init_image.convert("RGB")).astype(np.float32) / 255.0
        mask_image = np.array(mask_image.convert("L")).astype(np.float32) / 255.0

        assert init_image.shape[0:1] == mask_image.shape[0:1]
        init_image[mask_image > 0.5] = -1.0  # Set as masked pixel
        init_image = np.expand_dims(init_image, 0).transpose(0, 3, 1, 2)
        init_image = torch.from_numpy(init_image)
        return init_image
