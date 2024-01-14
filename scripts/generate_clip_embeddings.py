from argparse import ArgumentParser
import tqdm

import h5py
import torch
import torch.nn as nn
import numpy as np 
import pandas as pd
from transformers import CLIPVisionModelWithProjection, AutoProcessor
from diffusers import StableUnCLIPImg2ImgPipeline


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('model size: {:.3f}MB'.format(size_all_mb))

def display_hdf5_structure(args):
    def h5_tree(val, pre=''):
        items = len(val)
        for key, val in val.items():
            items -= 1
            if items == 0:
                # the last item
                if type(val) == h5py._hl.group.Group:
                    print(pre + '└── ' + key)
                    h5_tree(val, pre+'    ')
                else:
                    print(pre + '└── ' + key + ' (%d)' % len(val))
            else:
                if type(val) == h5py._hl.group.Group:
                    print(pre + '├── ' + key)
                    h5_tree(val, pre+'│   ')
                else:
                    print(pre + '├── ' + key + ' (%d)' % len(val))

    with h5py.File(args.images, 'r') as hf:
        print(hf)
        h5_tree(hf)

def inverse_linear_layer(original_linear):
    # Calculate the inverse of the weight matrix
    inverse_weights = torch.Tensor(np.linalg.pinv(original_linear.weight.data))

    # Create the inverse linear layer
    inverse_linear = nn.Linear(in_features=original_linear.out_features, 
                                out_features=original_linear.in_features, 
                                bias=original_linear.bias is not None)
    
    # Set the inverse weights and bias
    inverse_linear.weight.data = inverse_weights
    if original_linear.bias is not None:
        inverse_linear.bias.data = -torch.matmul(inverse_weights, original_linear.bias.data)

    return inverse_linear


def main(args):
    pipe = StableUnCLIPImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-2-1-unclip", torch_dtype=torch.float16, variation="fp16"
    )
    model = pipe.image_encoder #CLIPVisionModelWithProjection.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    processor = pipe.feature_extractor #AutoProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    print(f"params number: {count_parameters(model)}")
    model_size(model)

    with h5py.File(args.images, 'r') as f:
        image_count = len(f['imgBrick'])
        data = f['imgBrick'][:]
        print(f"Shape of the image dataset: {data.shape}")

    embeddings = torch.empty((image_count, model.config.projection_dim), dtype=torch.float32, device='cpu')

    model.to(args.device)
    model.eval()

    print("Generating embeddings...")
    with torch.no_grad():
        for i in tqdm.tqdm(range(0, image_count, args.batch_size), total=image_count // args.batch_size + 1):
            
            batch = data[i: i + args.batch_size]
            # print(batch.shape)

            inputs = processor(images=batch, return_tensors="pt", padding=True)["pixel_values"]
            # print(inputs.shape)

            # print(inputs.shape)
            inputs_gpu = inputs.to(args.device)
            outputs = model(pixel_values=inputs_gpu)
            embeds = outputs.image_embeds.to('cpu')
            embeddings[i: i + args.batch_size] = embeds

            del batch, inputs, inputs_gpu, outputs, embeds
            torch.cuda.empty_cache()
            # print(embeds.shape)
            # break 

    torch.save(embeddings, args.output)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images', type=str, help='path to hdf5 file containing images')
    parser.add_argument('--output', type=str, help='path to pt file to save embeddings to')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(args)

    
