from PIL import Image
import torch
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, UniPCMultistepScheduler
import matplotlib.pyplot as plt
from masks import MaskGenerator
import wandb

class InpaintingPipeline:
    def __init__(self):
        self.mask_generator = MaskGenerator()


    def inpainting(self, image_path, mask_text_prompt, final_text_prompt, save_results=True):

        run = wandb.init(project="visionbrush-inpainting")

        """Run the inpainting pipeline."""
        # Step 1: Generate masks
        masks, image_pil = self.mask_generator.segmentation_model(image_path, mask_text_prompt)

        # Step 2: Prepare masks for inpainting
        init_image, mask_image = self.mask_generator.inpainting_masks(masks, image_pil)

        # Step 3: Prepare the control image
        control_image = self.mask_generator.make_inpaint_condition(init_image, mask_image)

        # Step 4: Load the inpainting pipeline
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float32, use_safetensors=True
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float32, use_safetensors=True
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        #pipe.enable_model_cpu_offload()
        pipe.to("cpu")

        with torch.no_grad():
            # Step 5: Perform inpainting
            output = pipe(
                final_text_prompt,  # Text prompt for inpainting
                num_inference_steps=20,
                eta=1.0,
                image=init_image,
                mask_image=mask_image,
                control_image=control_image,
            ).images[0]
        # Define W&B Table to store generations
        columns = ["mask_text_prompt", "final_text_prompt"]
        table = wandb.Table(columns=columns)
        table.add_data(mask_text_prompt, final_text_prompt)
                # Log text separately in a table
        run.log({"prompts": table})
        # Log images in W&B
        run.log({
            "Original Image": wandb.Image(image_pil),
            "Masked Image": wandb.Image(mask_image),
            "Inpainted Image": wandb.Image(output)
        })

        # Step 6: Save and visualize results
        if save_results:
            # Debugging: Print image types and shapes
            print(f"init_image type: {type(init_image)}, mode: {init_image.mode}")
            print(f"mask_image type: {type(mask_image)}, mode: {mask_image.mode}")
            print(f"output type: {type(output)}, mode: {output.mode}")

            # Save the images
            init_image.save("test_images/original_image.png")
            mask_image.save("test_images/masked_image.png")
            output.save("test_images/inpainted_image.png")

            # Visualize the results
            #self.visualize_results(init_image, mask_image, output)


# Example usage
if __name__ == "__main__":
    image_path = "./assets/car.jpeg"
    mask_text_prompt = "wheels"
    final_text_prompt = "meatballs"

    pipeline = InpaintingPipeline()
    pipeline.inpainting(image_path, mask_text_prompt, final_text_prompt, save_results=True)