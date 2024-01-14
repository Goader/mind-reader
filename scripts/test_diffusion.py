import numpy as np 
import torch
import h5py 


def get_images_end_embeddings(images_path, embeddings_path) -> list[tuple[np.ndarray, torch.Tensor]]:
    n = 10
    with h5py.File(images_path, 'r') as f:
        data = f['imgBrick'][:n]
        print(f"Shape of the image dataset: {data.shape}")

    embeddings = torch.load(embeddings_path)[:n]
    
    return list(zip(data, embeddings))


def unclip():
    from PIL import Image
    import torch
    from diffusers import UnCLIPScheduler, DDPMScheduler, StableUnCLIPPipeline, StableUnCLIPImg2ImgPipeline
    from diffusers.models import PriorTransformer
    from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection, AutoProcessor

    images_path = "data/nsd_stimuli.hdf5"
    embeddings_path = "data/clip_embeddings.pt"
    combined = get_images_end_embeddings(images_path, embeddings_path)
    print(combined[0][1].shape)

    prior_text_model_id = "stabilityai/stable-diffusion-2-1-unclip"
    # prior_tokenizer = CLIPTokenizer.from_pretrained(prior_text_model_id)
    # prior_text_model = CLIPTextModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=data_type)

    # image_preprocessor = AutoProcessor.from_pretrained(prior_text_model_id)
    # prior_vision_model = CLIPVisionModelWithProjection.from_pretrained(prior_text_model_id, torch_dtype=torch.float16)


    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip"
    )

    pipe = pipe.to("cuda")
    prompt = "photorealistic; High quality"
    generated = []

    with torch.no_grad():
        for i, (img, embedding) in enumerate(combined):
            img_pred = pipe(image_embeds=embedding.unsqueeze(0).to("cuda"), prompt=prompt, num_inference_steps=50).images[0]

            img = Image.fromarray(img)
            img.save(f"img{i}.png")
            img_pred.save(f"img{i}_pred.png")


unclip()
