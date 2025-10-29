from tqdm import tqdm
from omegaconf import OmegaConf
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import cv2

from datasets.transforms import get_transforms
from extend_sam import get_model
from extend_sam.utils import get_numpy_from_tensor, check_folder

def load_model(model, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    model.eval()
    print(f'model loaded in {device}...')
    return model, device

def hex_to_rgb(hex_color):
    """ Convert hex keys to RGB tuple keys """
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def convert_anns_to_labels(ann_pred, id2color):
    """ Converting annotation prediction to label_img are cleaned using cv2 followed by
    creating the class_mask of the annotations to use for training """
    h, w = ann_pred.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, rgb_color in id2color.items():
        matches = ann_pred == class_id
        # print(class_id, rgb_color, np.sum(matches))
        color_mask[matches] = np.array(rgb_color).astype(np.uint8)
    # background is left black
    return color_mask

def generate_grid_points(img, points_per_side=32):
    # may not be useful as typically all points for point_prompt are considered
    # as denoting one single object. Our objective is to detect multiple objects.

    _, _, h, w = img.shape
    y_coords = np.linspace(0, h - 1, points_per_side).astype(int)
    x_coords = np.linspace(0, w - 1, points_per_side).astype(int)
    points = np.stack(np.meshgrid(x_coords, y_coords), axis=-1).reshape(-1, 2) # [N, 2]

    # point labels:
    # 0 - background, telling SAM to assume there might not be an object
    # 1 - foreground, telling SAM to assume there might be an object

    tensor_points = torch.tensor(points, dtype=torch.float32, device=img.device)  # shape [N, 2]
    tensor_labels = torch.ones(tensor_points.shape[0], dtype=torch.int64, device=img.device)  # shape [N]

    return tensor_points.unsqueeze(0), tensor_labels.unsqueeze(0) # batch dim


if __name__ == '__main__':
    # Load inference configs
    config_file_path = "./config/inference.yaml"
    check_folder(config_file_path)
    config = OmegaConf.load(config_file_path)
    infer_cfg = config.inference

    # Check file and folders
    dataset_path = Path(infer_cfg.dataset.input_dir)
    images_path = dataset_path / 'img' / 'infer'
    check_folder(images_path, is_folder=True)
    output_path = Path(infer_cfg.dataset.output_dir)
    check_folder(output_path, is_folder=True)
    metainfo_path = dataset_path / 'metainfo.yaml'
    assert metainfo_path.exists(), 'need same metainfo.yaml used for training to compile class names in order'

    # Check correctness and consistency in class detail
    metainfo = OmegaConf.load(metainfo_path)
    class_names = metainfo['class_names']
    assert infer_cfg.dataset.label2color is not None, 'create label2color with hexadecimal color for each class_name'
    label2color = infer_cfg.dataset.label2color
    assert set(label2color.keys()) == set(class_names), 'missing class_name in inference.dataset.label2color'
    id2color = {0: (0,0,0)} # background set to black
    for id, class_name in enumerate(class_names):
        color = label2color[class_name]
        id2color[id+1] = hex_to_rgb(color)
    # print(id2color)

    # Initialize transform
    assert infer_cfg.dataset.transforms is not None, 'required'
    transform = get_transforms(infer_cfg.dataset.transforms)

    # Load trained model
    check_folder(infer_cfg.model.model_path)
    model = get_model(model_name=infer_cfg.model.sam_name, **infer_cfg.model.params)
    model, device = load_model(model, infer_cfg.model.model_path)

    # Run inference
    valid_suffix = ['.png', '.jpg']
    img_info = [(path.stem, path.suffix) for path in sorted(images_path.iterdir()) if path.suffix in valid_suffix]
    for stem, suffix in tqdm(img_info):
        img_filename = f"{stem}{suffix}"
        out_filename = f"{stem}.jpg"
        img_path = images_path / img_filename
        out_path = output_path / out_filename
        img = Image.open(img_path)
        image = transform(img).unsqueeze(0)  # add batch dimension
        with torch.inference_mode():
            mask_pred, _ = model(image.to(device))
            mask_pred = mask_pred.squeeze(0)
        prediction = get_numpy_from_tensor(torch.argmax(mask_pred, dim=0)) # mask_logits to mask_classes
        ann_pred = cv2.resize(prediction, img.size, interpolation=cv2.INTER_NEAREST)
        color_mask = convert_anns_to_labels(ann_pred, id2color)
        Image.fromarray(color_mask, mode='RGB').save(out_path)

    del model







