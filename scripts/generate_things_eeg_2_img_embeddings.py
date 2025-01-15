import torch
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import pickle
import argparse

from dreamsim import dreamsim, PerceptualModel
from dreamsim.config import dreamsim_args

def get_embeddings(encoder, name, images, root):
  embeddings = []
  for img in tqdm(images):
    img = preprocess(img).to(device)
    embeddings.append(encoder.embed(img).detach().cpu())
  with open(os.path.join(root, f"{name}_embeds.pkl"), "wb") as f:
    pickle.dump(embeddings, f)


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
    img_parent_dir  = os.path.join(data_path, 'images')
    img_parent_save_dir = os.path.join(save_path)
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

    train_embeddings = []
    for item in range(16540):
        img_file = os.path.join(img_parent_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img = Image.open(img_file).convert('RGB')
        img = preprocess(img).to(device)

        with torch.no_grad():
            e = model.embed(img).detach().cpu().numpy()

        # train_embeddings.append(e)
        if args.human_aligned:
            np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy"), e)
        else:
            np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}_noalign.npy"), e)
        if item % 1000 == 0:
            print(f"{item} items out of 16540 done")
            print(f"e.shape = {e.shape}")
    # train_embeddings = np.array(train_embeddings).squeeze()
    # print(f"train_embeddings.shape = {train_embeddings.shape}")
    # if args.human_aligned:
    #     np.save(os.path.join(img_parent_save_dir, f"train_{args.model_name}_{args.model_type}.npy"), train_embeddings)
    # else:
    #     np.save(os.path.join(img_parent_save_dir, f"train_{args.model_name}_{args.model_type}_noalign.npy"), train_embeddings)

    print("Start Embedidng Test Images")
    test_embeddings = []
    for item in range(200):
        img_file = os.path.join(img_parent_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img = Image.open(img_file).convert('RGB')
        img = preprocess(img).to(device)

        with torch.no_grad():
            e = model.embed(img).detach().cpu().numpy()

        # test_embeddings.append(e)
        if args.human_aligned:
            np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}.npy"), e)
        else:
            np.save(img_save_file.replace(".jpg", f"_{args.model_name}_{args.model_type}_noalign.npy"), e)

    # test_embeddings = np.array(test_embeddings).squeeze()
    # print(f"test_embeddings.shape = {test_embeddings.shape}")
    # if args.human_aligned:
    #     np.save(os.path.join(img_parent_save_dir, f"test_{args.model_name}_{args.model_type}.npy"), test_embeddings)
    # else:
    #     np.save(os.path.join(img_parent_save_dir, f"test_{args.model_name}_{args.model_type}_noalign.npy"), test_embeddings)



