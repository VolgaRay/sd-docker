from diffusers.quantizers.quantization_config import BitsAndBytesConfig
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

model_id = "stabilityai/stable-diffusion-3.5-large-turbo"

nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_nf4 = SD3Transformer2DModel.from_pretrained(
    model_id,
    subfolder="transformer",
    quantization_config=nf4_config,
    torch_dtype=torch.bfloat16
)

pipeline = DiffusionPipeline.from_pretrained(
    model_id, 
    transformer=model_nf4,
    torch_dtype=torch.bfloat16
)
# pipeline.enable_model_cpu_offload()

pipeline = pipeline.to("cuda")

prompt = "Orange cat astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
image = pipeline(  # type: ignore
    prompt=prompt,
    num_inference_steps=4,
    guidance_scale=0.0,
    max_sequence_length=512,
).images[0]

image.save("./Astronaut.png")
