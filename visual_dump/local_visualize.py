import numpy as np
from visual_utils.open3d_vis_utils import draw_scenes

data = np.load("sample_000.npz")
points = data['points']
pred_boxes = data['pred_boxes']
pred_scores = data['pred_scores']
pred_labels = data['pred_labels']

draw_scenes(points=points, ref_boxes=pred_boxes, ref_scores=pred_scores, ref_labels=pred_labels)
