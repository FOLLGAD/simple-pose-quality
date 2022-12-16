from PIL import Image
from mmpose.apis import inference_top_down_pose_model, init_pose_model, process_mmdet_results
from mmdet.apis import inference_detector, init_detector
from typing import List
import numpy as np
import torch
from fastai.tabular.all import *
import pandas as pd
import glob


def create_pose_model(device: torch.device):
  pose_config = 'mmcv-configs/pose.py'
  pose_checkpoint = 'https://download.openmmlab.com/mmpose/top_down/resnet/res50_coco_256x192-ec54d7f3_20200709.pth'
  pose_model = init_pose_model(pose_config, pose_checkpoint, device=device if device.type != 'mps' else None)
  return pose_model


def get_poses(image: Image, pose_model) -> List[dict]:
  pose_results, _ = inference_top_down_pose_model(pose_model, np.asarray(image))

  poses = []
  for pose_result in pose_results:
    keypoints = {}
    for index, point in enumerate(pose_result['keypoints']):
      keypoint_name = pose_model.cfg.dataset_info.keypoint_info[index].name
      keypoints[keypoint_name + "_x"] = np.float32(point[2])
      if (len(point) > 3):
        keypoints[keypoint_name + "_y"] = np.float32( point[3])
    poses.append(keypoints)

  return poses[0]

d = torch.device("cpu")
pose_model = create_pose_model(d)
# open all imags in the dataset/good folder
imgs = [Image.open(f) for f in glob.glob("dataset/good/*.jpg")]
goodposes = [get_poses(img, pose_model) for img in imgs]
imgs = [Image.open(f) for f in glob.glob("dataset/bad/*.jpg")]
badposes = [get_poses(img, pose_model) for img in imgs]


# create a dataframe with the keypoints
df = pd.DataFrame(goodposes)
df['label'] = 1
df2 = pd.DataFrame(badposes)
df2['label'] = 0
df = df.append(df2)
df = df.fillna(0)

cols = list(df.columns)[:-1]
print(cols)

dls = TabularDataLoaders.from_df(df, path='.',
    cat_names=[],
    cont_names=cols,
    y_block = CategoryBlock,
    y_names='label',
    procs=[Categorify,FillMissing, Normalize], bs=16)
learn = tabular_learner(dls, metrics=accuracy)
learn.fit_one_cycle(5)

print(dls)

dls.show_batch()

testing  = pd.DataFrame([get_poses(Image.open("poses/bad.jpeg"), pose_model)])
_,c3, _ = learn.predict(testing.iloc[0])
testing  = pd.DataFrame([get_poses(Image.open("poses/wrongorient.jpg"), pose_model)])
_,c2, _ = learn.predict(testing.iloc[0])
testing  = pd.DataFrame([get_poses(Image.open("poses/behind.jpg"), pose_model)])
_,c4, _ = learn.predict(testing.iloc[0])
testing  = pd.DataFrame([get_poses(Image.open("poses/goodman.jpg"), pose_model)])
_,c5, _ = learn.predict(testing.iloc[0])

print("good", c5)
print("bad", c2)
print("good2", c3)
print("behind", c4)