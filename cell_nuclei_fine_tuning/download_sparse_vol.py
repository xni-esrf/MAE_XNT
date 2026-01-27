import webknossos as wk
from pathlib import Path
import numpy as np
import os
import tifffile
import h5py
import json
import sys

import scipy.ndimage

output_dir = sys.argv[1]

URL = "https://webknossos.crick.ac.uk/annotations/66b693eb010000f900161abc/"
annotation = wk.Annotation.download(URL)
data = wk.Annotation.open_as_remote_dataset(URL)
bboxes = annotation.user_bounding_boxes
all_point_coord = [n.position.to_tuple() for t in annotation.skeleton.flattened_trees() for n in t.nodes]
# Rectify the missplaced point
all_point_coord.remove((6160, 7620, 4102))
all_point_coord.append((6160, 7620, 4100))
added_points = []

for bbox in bboxes:
    cur_z_limit = [bbox.topleft[0], bbox.bottomright[0]]
    cur_x_limit = [bbox.topleft[1], bbox.bottomright[1]]
    cur_y_limit = [bbox.topleft[2], bbox.bottomright[2]]

    new_sparse_test_seg = np.zeros([bbox.size[0],bbox.size[1],bbox.size[2]], dtype=np.float32)

    # Add gt point with coordinates within the processed volume
    for point_coord in all_point_coord:
        if point_coord[0] >= cur_z_limit[0]-1 and point_coord[0] <= cur_z_limit[1]+1:
            if point_coord[1] >= cur_x_limit[0]-1 and point_coord[1] <= cur_x_limit[1]+1:
                if point_coord[2] >= cur_y_limit[0]-1 and point_coord[2] <= cur_y_limit[1]+1:
                    relative_point_coord = [point_coord[0]-cur_z_limit[0]-1,point_coord[1]-cur_x_limit[0]-1,point_coord[2]-cur_y_limit[0]-1]
                    new_sparse_test_seg[relative_point_coord[0],relative_point_coord[1],relative_point_coord[2]] = 1
                    added_points.append(point_coord)      

    # Add the 8um tolerance margin
    new_sparse_test_seg = np.pad(new_sparse_test_seg, pad_width=((80, 80), (80, 80), (80, 80)), mode='constant', constant_values=0)
    bbox = bbox.padded_with_margins([80,80,80],[80,80,80])
    color_data = data.get_layer("color").get_mag(2).read(absolute_bounding_box=bbox.with_topleft_xyz(bbox.topleft_xyz // 2 * 2).align_with_mag(wk.Vec3Int(2,2,2)))

    # Normalize data and remove conversion outliers
    color_data = np.squeeze(color_data)
    color_data = color_data.astype(np.float64)
    mean_color_data = color_data[40:290,40:290,40:290].mean()
    std_color_data = color_data[40:290,40:290,40:290].std()
    color_data = (color_data-mean_color_data)/std_color_data
    color_data = np.clip(color_data,-10,10)

    # Go back to the original 100 nm resolution
    color_data = scipy.ndimage.zoom(color_data, zoom=(2, 2, 2), order=3)

    os.mkdir(os.path.join(output_dir,bbox.name))
    
    color_data = color_data.astype(np.float32)
    with h5py.File(os.path.join(output_dir,bbox.name,"raw.h5"), 'w', rdcc_nbytes=(10*1024*1024), rdcc_w0=0.5, rdcc_nslots=10009) as hf:
        hf.create_dataset(bbox.name, data=color_data, chunks=(50,50,50))

    with h5py.File(os.path.join(output_dir,bbox.name,"annotations.h5"), 'w', rdcc_nbytes=(10*1024*1024), rdcc_w0=0.5, rdcc_nslots=10009) as hf:
        hf.create_dataset(bbox.name, data=new_sparse_test_seg, chunks=(50,50,50))

    tifffile.imwrite(os.path.join(output_dir,bbox.name,"raw.tiff"), color_data)
    tifffile.imwrite(os.path.join(output_dir,bbox.name,"annotations.tiff"), new_sparse_test_seg)

    # Build volume metadata
    cur_metadata_dict = {}
    cur_metadata_dict["volume_path"] = os.path.join(output_dir,bbox.name,"raw.h5")
    cur_metadata_dict["annot_path"] = os.path.join(output_dir,bbox.name,"annotations.h5")
    cur_metadata_dict["volume.shape"] = color_data.shape

    metadata_dir_path = "metadata/C432_sparse"
    with open(os.path.join(metadata_dir_path,bbox.name+".json"), 'w') as json_file:
        json.dump(cur_metadata_dict, json_file)

