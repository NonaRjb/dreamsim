import sys

sys.path.append('/proj/rep-learning-robotics/users/x_nonra/eeg_asif_img/src')

import torch
import argparse
import os
import numpy as np
from PIL import Image
from models.image_architectures import CLIP_IMG, DINO, DINOv2, OpenCLIP


model_name_map = {
    'CLIP': {
        'ViT-B32': 'openai/clip-vit-base-patch32',
        'ViT-B16': 'openai/clip-vit-base-patch16',
        'ViT-L14': 'openai/clip-vit-large-patch14',
    },
    'DINO': {
        'ViT-B16': 'facebook/dino-vitb16',
        'ViT-B8': 'facebook/dino-vitb8',
    },
    'DINOv2': {
        'ViT-B14': 'facebook/dinov2-base',
        'ViT-L14': 'facebook/dinov2-large',
    },
    'OpenCLIP': {
        'ViT-L14_laion2b': 'hf-hub:laion/CLIP-ViT-L-14-laion2B-s32B-b82K',
        'ViT-L14_laion400m': 'hf-hub:timm/vit_large_patch14_clip_224.laion400m_e32',
        'ViT-B32_laion400m': 'hf-hub:timm/vit_base_patch32_clip_224.laion400m_e31',
        'RN50': 'hf-hub:timm/resnet50_clip.openai'
    }
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_path",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/image_embeddings"
    )
    parser.add_argument(
        "--data_path",
        default="/proj/rep-learning-robotics/users/x_nonra/alignvis/data/things_eeg_2/images"
    )
    parser.add_argument("--model_type", type=str, default="CLIP")
    parser.add_argument("--model_name", type=str, default="ViT-B32")

    return parser.parse_args()

def get_model(model_type, model_name):
    if model_type == "CLIP":
        model = CLIP_IMG(model_name=model_name_map[model_type][model_name], alr_preprocessed=False)
    elif model_type == "DINO":
        model = DINO(model_name=model_name_map[model_type][model_name], alr_preprocessed=False)
    elif model_type == "DINOv2":
        model = DINOv2(model_name=model_name_map[model_type][model_name], alr_preprocessed=False)
    elif model_type == "OpenCLIP":
        model = OpenCLIP(model_name=model_name_map[model_type][model_name], alr_preprocessed=False)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return model


if __name__ == "__main__":

    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model(args.model_type, args.model_name)

    data_path = args.data_path
    save_path = data_path if args.save_path is None else args.save_path
    img_parent_dir  = data_path
    img_parent_save_dir = save_path
    img_metadata = np.load(os.path.join(img_parent_dir, 'image_metadata.npy'), allow_pickle=True).item()

    train_embeddings = []
    for item in range(16540):
        img_file = os.path.join(img_parent_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'training_images', 
                        img_metadata['train_img_concepts'][item], img_metadata['train_img_files'][item])
        img = Image.open(img_file).convert('RGB')

        with torch.no_grad():
            e = model(img).detach().cpu().numpy()

        # train_embeddings.append(e)
        np.save(img_save_file.replace(".jpg", f"_{args.model_type}_{args.model_name}_noalign.npy"), e)
        if item % 1000 == 0:
            print(f"{item} items out of 16540 done")
            print(f"e.shape = {e.shape}")

    print("Start Embedidng Test Images")
    test_embeddings = []
    for item in range(200):
        img_file = os.path.join(img_parent_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img_save_file = os.path.join(img_parent_save_dir, 'test_images', 
                        img_metadata['test_img_concepts'][item], img_metadata['test_img_files'][item])
        img = Image.open(img_file).convert('RGB')

        with torch.no_grad():
            e = model(img).detach().cpu().numpy()

        # test_embeddings.append(e)
        np.save(img_save_file.replace(".jpg", f"_{args.model_type}_{args.model_name}_noalign.npy"), e)


