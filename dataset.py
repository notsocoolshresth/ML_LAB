"""
Landslide4Sense Dataset Loader with Spectral Index Features.
Supports the original L4S format:
 TrainData/img/*.h5, TrainData/mask/*.h5
 TestData/img/*.h5
 ValidData/img/*.h5
"""

import os
import glob
import re
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset
import albumentations as A


def _find_subdir_case_insensitive(parent_dir, expected_name):
 """Return the matching child directory path, ignoring case."""
 if not os.path.isdir(parent_dir):
  return None
 for name in os.listdir(parent_dir):
  if name.lower() == expected_name.lower():
   candidate = os.path.join(parent_dir, name)
   if os.path.isdir(candidate):
    return candidate
 return None


def _has_h5_images(split_dir):
 """Check whether split_dir/img contains at least one .h5 file."""
 img_dir = _find_subdir_case_insensitive(split_dir, "img")
 if img_dir is None:
  return False
 return len(glob.glob(os.path.join(img_dir, "*.h5"))) > 0


def _is_l4s_h5_root(root_dir):
 """True when root has a supported Landslide4Sense h5 split layout.

 Supported layouts:
 - legacy: TrainData/img/*.h5, TestData/img/*.h5
 - huggingface: images/train/*.h5, images/test/*.h5
 """
 # Legacy layout
 train_dir = _find_subdir_case_insensitive(root_dir, "TrainData")
 test_dir = _find_subdir_case_insensitive(root_dir, "TestData")
 if (
  train_dir is not None
  and test_dir is not None
  and _has_h5_images(train_dir)
  and _has_h5_images(test_dir)
 ):
  return True

 # Hugging Face layout
 images_dir = _find_subdir_case_insensitive(root_dir, "images")
 if images_dir is None:
  return False

 hf_train = _find_subdir_case_insensitive(images_dir, "train")
 hf_test = _find_subdir_case_insensitive(images_dir, "test")
 if hf_train is None or hf_test is None:
  return False

 return (
  len(glob.glob(os.path.join(hf_train, "*.h5"))) > 0
  and len(glob.glob(os.path.join(hf_test, "*.h5"))) > 0
 )


def resolve_l4s_data_dir(data_dir):
 """Resolve a usable Landslide4Sense h5 root directory.

 Tries:
 1) provided path
 2) direct child folders of provided path
 3) sibling folders of provided path
 """
 requested = os.path.abspath(os.path.expanduser(data_dir))
 candidates = []

 def _add(path):
  if path and path not in candidates:
   candidates.append(path)

 _add(requested)

 if os.path.isdir(requested):
  for name in os.listdir(requested):
    child = os.path.join(requested, name)
    if os.path.isdir(child):
     _add(child)

 parent = os.path.dirname(requested)
 if os.path.isdir(parent):
  for name in os.listdir(parent):
    sibling = os.path.join(parent, name)
    if os.path.isdir(sibling):
     _add(sibling)

 for candidate in candidates:
  if _is_l4s_h5_root(candidate):
   if candidate != requested:
    print(f"Resolved data_dir: '{data_dir}' -> '{candidate}'")
   return candidate

 raise FileNotFoundError(
  "Could not find Landslide4Sense h5 dataset layout (TrainData/img/*.h5, "
  f"TestData/img/*.h5) from '{data_dir}'."
 )


def compute_spectral_indices(img):
 eps = 1e-8
 red = img[:, :, 3].astype(np.float32)
 nir = img[:, :, 7].astype(np.float32)
 green = img[:, :, 2].astype(np.float32)
 swir1 = img[:, :, 10].astype(np.float32)
 ndvi = (nir - red) / (nir + red + eps)
 ndwi = (green - nir) / (green + nir + eps)
 ndbi = (swir1 - nir) / (swir1 + nir + eps)
 bsi = ((swir1 + red) - (nir + green)) / ((swir1 + red) + (nir + green) + eps)
 return np.stack([ndvi, ndwi, ndbi, bsi], axis=-1)


def _find_h5_key(path):
 """Safely read the first key from an h5 file."""
 try:
  with h5py.File(path, "r") as f:
   keys = list(f.keys())
   if len(keys) > 0:
    return keys[0]
 except Exception:
  pass
 return None


def _extract_numeric_id(path):
 """Extract trailing numeric id from filenames like image_123.h5 / mask_123.h5."""
 name = os.path.basename(path)
 match = re.search(r"_(\d+)\.h5$", name)
 return int(match.group(1)) if match else None


def _find_mask_dir(split_dir):
 """Find the mask subdirectory, trying mask/ then test/."""
 # Hugging Face layout: masks can be directly inside split_dir.
 direct_files = sorted(glob.glob(os.path.join(split_dir, "*.h5")))
 if len(direct_files) > 0:
  for fp in direct_files:
   key = _find_h5_key(fp)
   if key is not None:
    return split_dir, direct_files, key

 for subdir in ["mask", "test"]:
  d = os.path.join(split_dir, subdir)
  if os.path.isdir(d):
   files = sorted(glob.glob(os.path.join(d, "*.h5")))
   if len(files) > 0:
    # Verify at least one file has actual data
    for fp in files:
     key = _find_h5_key(fp)
     if key is not None:
      return d, files, key
 return None, [], None


class LandslideDataset(Dataset):
 def __init__(self, data_dir, split="train", transform=None, use_indices=True):
  self.transform = transform
  self.use_indices = use_indices
  self.split = split

  split_map_legacy = {"train": "TrainData", "validation": "ValidData", "test": "TestData"}
  split_map_hf = {"train": "train", "validation": "validation", "test": "test"}

  # Resolve image and annotation split directories for supported layouts.
  images_root = _find_subdir_case_insensitive(data_dir, "images")
  ann_root = _find_subdir_case_insensitive(data_dir, "annotations")

  if images_root is not None:
   img_split_dir = _find_subdir_case_insensitive(images_root, split_map_hf[split])
   if img_split_dir is None:
    raise FileNotFoundError(
     f"Image split folder '{split_map_hf[split]}' not found under {images_root}"
    )
   mask_split_dir = None
   if ann_root is not None:
    mask_split_dir = _find_subdir_case_insensitive(ann_root, split_map_hf[split])
  else:
   split_dir = _find_subdir_case_insensitive(data_dir, split_map_legacy[split])
   if split_dir is None:
    raise FileNotFoundError(f"Split folder '{split_map_legacy[split]}' not found in {data_dir}")
   img_split_dir = split_dir
   mask_split_dir = split_dir

  # Images: either split_dir/*.h5 (HF) or split_dir/img/*.h5 (legacy)
  direct_img_files = glob.glob(os.path.join(img_split_dir, "*.h5"))
  if len(direct_img_files) > 0:
   img_dir = img_split_dir
  else:
   img_dir = _find_subdir_case_insensitive(img_split_dir, "img")
   if img_dir is None:
    raise FileNotFoundError(f"Image folder not found in {img_split_dir}")

  self.img_paths = sorted(
   glob.glob(os.path.join(img_dir, "*.h5")),
   key=lambda p: (_extract_numeric_id(p) is None, _extract_numeric_id(p), p),
  )
  if len(self.img_paths) == 0:
   raise FileNotFoundError(f"No .h5 files found in {img_dir}")

  self.img_key = _find_h5_key(self.img_paths[0])
  if self.img_key is None:
   raise ValueError(f"First image file is empty: {self.img_paths[0]}")

  # Masks
  mask_search_dir = mask_split_dir if mask_split_dir is not None else img_split_dir
  mask_dir, raw_mask_paths, self.mask_key = _find_mask_dir(mask_search_dir)
  self.mask_map = {}
  skipped_masks = 0

  if mask_dir is not None and len(raw_mask_paths) > 0:
   for mp in sorted(raw_mask_paths, key=lambda p: (_extract_numeric_id(p) is None, _extract_numeric_id(p), p)):
    mask_id = _extract_numeric_id(mp)
    if mask_id is None:
     skipped_masks += 1
     continue
    key = _find_h5_key(mp)
    if key is None:
     skipped_masks += 1
     continue
    self.mask_map[mask_id] = (mp, key)

  self.has_masks = len(self.mask_map) > 0
  matched = sum(1 for p in self.img_paths if _extract_numeric_id(p) in self.mask_map)

  print(
   f"[{split}] {len(self.img_paths)} images (key='{self.img_key}'), "
   f"masks={'yes' if self.has_masks else 'no'} "
   f"(valid={len(self.mask_map)}, matched={matched}, skipped_invalid={skipped_masks})"
  )

 def __len__(self):
  return len(self.img_paths)

 def _load_sample(self, idx):
  img_path = self.img_paths[idx]
  with h5py.File(img_path, "r") as f:
   img = f[self.img_key][:].astype(np.float32)

  img_id = _extract_numeric_id(img_path)
  if self.has_masks and img_id in self.mask_map:
   mask_path, key = self.mask_map[img_id]
   try:
    with h5py.File(mask_path, "r") as f:
     mask = f[key][:].astype(np.float32)
   except Exception:
    mask = np.zeros((128, 128), dtype=np.float32)
  else:
   mask = np.zeros((128, 128), dtype=np.float32)

  # Ensure HWC
  if img.ndim == 3 and img.shape[0] == 14:
   img = img.transpose(1, 2, 0)

  if mask.ndim == 3:
   mask = mask.squeeze()

  mask = (mask > 0).astype(np.float32)

  return img, mask

 def __getitem__(self, idx):
  img, mask = self._load_sample(idx)

  if self.use_indices:
   indices = compute_spectral_indices(img)
   img = np.concatenate([img, indices], axis=-1)

  if self.transform is not None:
   transformed = self.transform(image=img, mask=mask)
   img = transformed["image"]
   mask = transformed["mask"]

  img = torch.from_numpy(img.transpose(2, 0, 1).copy())
  mask = torch.from_numpy(mask.copy()).unsqueeze(0)
  return img, mask


class _SubsetWithTransform(Dataset):
 def __init__(self, parent, indices, transform):
  self.parent = parent
  self.indices = list(indices)
  self.transform = transform

 def __len__(self):
  return len(self.indices)

 def __getitem__(self, idx):
  real_idx = self.indices[idx]
  img, mask = self.parent._load_sample(real_idx)

  if self.parent.use_indices:
   indices_feat = compute_spectral_indices(img)
   img = np.concatenate([img, indices_feat], axis=-1)

  if self.transform is not None:
   transformed = self.transform(image=img, mask=mask)
   img = transformed["image"]
   mask = transformed["mask"]

  img = torch.from_numpy(img.transpose(2, 0, 1).copy())
  mask = torch.from_numpy(mask.copy()).unsqueeze(0)
  return img, mask


def get_train_transform():
 return A.Compose([
  A.HorizontalFlip(p=0.5),
  A.VerticalFlip(p=0.5),
  A.RandomRotate90(p=0.5),
  A.Affine(translate_percent=0.1, scale=(0.85, 1.15), rotate=(-30, 30), border_mode=0, p=0.5),
  A.GaussianBlur(blur_limit=(3, 5), p=0.2),
 ])


def get_val_transform():
 return None


def get_dataloaders(data_dir, batch_size=16, num_workers=4, val_split_ratio=0.15,
 use_indices=True, seed=42):

 data_dir = resolve_l4s_data_dir(data_dir)

 # Keep these args for backward compatibility; validation now uses ValidData directly.
 _ = val_split_ratio, seed

 train_dataset = LandslideDataset(
  data_dir, split="train", transform=get_train_transform(), use_indices=use_indices
 )
 val_dataset = LandslideDataset(
  data_dir, split="validation", transform=get_val_transform(), use_indices=use_indices
 )

 train_loader = torch.utils.data.DataLoader(
  train_dataset, batch_size=batch_size, shuffle=True,
  num_workers=num_workers, pin_memory=True, drop_last=True,
 )
 val_loader = torch.utils.data.DataLoader(
  val_dataset, batch_size=batch_size, shuffle=False,
  num_workers=num_workers, pin_memory=True,
 )

 test_dataset = LandslideDataset(data_dir, split="test", transform=get_val_transform(), use_indices=use_indices)
 test_loader = torch.utils.data.DataLoader(
  test_dataset, batch_size=batch_size, shuffle=False,
  num_workers=num_workers, pin_memory=True,
 )

 return train_loader, val_loader, test_loader
