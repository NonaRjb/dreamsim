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
    train_dir = 'images_meg'
    os.makedirs(os.path.join(save_path, train_dir), exist_ok=True)
    train_img_parent_dir = os.path.join(data_path, train_dir)
    train_concepts = os.listdir(train_img_parent_dir)
    train_concepts.sort()
    train_embeddings = []
    for n, concept in enumerate(train_concepts):
        train_img_files = os.listdir(os.path.join(train_img_parent_dir, concept))
        train_img_files.sort()
        for i, item in enumerate(train_img_files):
            img_file = os.path.join(train_img_parent_dir, concept, item)
            img = Image.open(img_file).convert('RGB')

            with torch.no_grad():
                e = model(img).detach().cpu().numpy()

            # train_embeddings.append(e)
            os.makedirs(os.path.join(save_path, train_dir, concept), exist_ok=True)
            np.save(os.path.join(save_path, train_dir, concept, item.replace(".jpg", f"_{args.model_type}_{args.model_name}_noalign.npy")), e)
        if n % 100 == 0:
            print(f"{n} concepts out of {len(train_concepts)} done")
            print(f"e.shape = {e.shape}")
    
    print("Start Embedidng Test Images")
    test_dir = 'images_test_meg'
    os.makedirs(os.path.join(save_path, test_dir), exist_ok=True)
    test_img_parent_dir = os.path.join(data_path, test_dir)
    test_img_files = os.listdir(test_img_parent_dir)
    test_img_files.sort()
    test_embeddings = []
    for item in test_img_files:
        img_file = os.path.join(test_img_parent_dir, item)
        img = Image.open(img_file).convert('RGB')

        with torch.no_grad():
            e = model(img).detach().cpu().numpy()

        # test_embeddings.append(e)
        np.save(os.path.join(save_path, test_dir, item.replace(".jpg", f"_{args.model_type}_{args.model_name}_noalign.npy")), e)

