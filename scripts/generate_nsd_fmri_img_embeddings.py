import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import pickle
import h5py
import argparse

from dreamsim import dreamsim, PerceptualModel
from dreamsim.config import dreamsim_args


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Image Embeddings")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the data directory")
    parser.add_argument("--save_path", type=str, default=None, help="Path to the output directory")
    parser.add_argument("--model_type", type=str, default="clip_vitb32", help="Type of model to use")
    parser.add_argument("--human_aligned", action="store_true", help="Use human aligned model")
    parser.add_argument("--model_name", type=str, default="dreamsim", help="Name of the method")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = dreamsim(pretrained=True, device=device, dreamsim_type=args.model_type, normalize_embeds=False)
    
    if not args.human_aligned:
        model_list = dreamsim_args['model_config'][args.model_type]['model_type'].split(",")
        model = PerceptualModel(**dreamsim_args['model_config'][args.model_type], device=device, load_dir="./models",
                                normalize_embeds=False, baseline=True)

    data_path = args.data_path
    save_path = data_path if args.save_path is None else args.save_path
    
    # npy_images = np.load(os.path.join(data_path, "all_subjects_unique_imagesnsd.npy"), mmap_mode='r')
    image_dataset = h5py.File(os.path.join(data_path, "nsd_stimuli.hdf5"), 'r')
    npy_images = image_dataset['imgBrick']

    train_embeddings = []
    for item in range(len(npy_images)):
        # Transpose to (224, 224, 3) for each image
        # image_transposed = np.transpose(npy_images[item].copy(), (1, 2, 0))
        # Convert the data type to uint8
        image_transposed = npy_images[item].copy().astype(np.uint8)
        # Convert to PIL Image
        img = Image.fromarray(image_transposed)
        # Preprocess the image
        img = preprocess(img).to(device)

        with torch.no_grad():
            e = model.embed(img).detach().cpu().numpy()

        train_embeddings.append(e)
        if item % 1000 == 0:
            print(f"{item} items out of {len(npy_images)} done")
            print(f"e.shape = {e.shape}")
        # if args.human_aligned:
        #     np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy"), e)
        # else:
        #     np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}_noalign.npy"), e)
        # if item % 1000 == 0:
        #     print(f"{item} items out of 16540 done")
        #     print(f"e.shape = {e.shape}")
    train_embeddings = np.array(train_embeddings).squeeze()
    print(f"train_embeddings.shape = {train_embeddings.shape}")
    if args.human_aligned:
        np.save(os.path.join(save_path, f"train_{args.model_name}_{args.model_type}.npy"), train_embeddings)
    else:
        np.save(os.path.join(save_path, f"train_{args.model_name}_{args.model_type}_noalign.npy"), train_embeddings)

    print("Done!!")