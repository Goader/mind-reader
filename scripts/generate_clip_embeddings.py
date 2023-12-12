from argparse import ArgumentParser
import tqdm

import h5py
import torch
import pandas as pd
from transformers import CLIPVisionModelWithProjection, AutoProcessor


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--images', type=str, help='path to hdf5 file containing images')
    parser.add_argument('--output', type=str, help='path to pt file to save embeddings to')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    with h5py.File(args.images, 'r') as f:
        image_count = len(f['images'])
    embeddings = torch.empty((image_count, model.config.projection_dim), dtype=torch.float32, device='cpu')

    model.to(args.device)
    model.eval()

    reader = enumerate(pd.read_hdf(args.images, iterator=True, chunksize=args.batch_size))
    for i, df_chunk in tqdm.tqdm(reader, total=int(args.batch_size / args.batch_size) + 1):
        start_idx = i * args.batch_size

        images = ...

        inputs = processor(images=images, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        embeds = outputs.image_embeds.to('cpu')
        embeddings[start_idx:start_idx+embeds.size(0)] = embeds

    torch.save(embeddings, args.output)

