import numpy as np
import cv2
import os

if __name__ == "__main__":
    residual_root_path = "/data1/MotionSeg3d_Dataset/data_odometry_velodyne/dataset/sequences/08"
    residual_id = [1, 2, 3, 4, 5, 6, 7, 8]
    # residual_id = [1, 3, 5, 7, 9, 11, 13, 15]
    # residual_id = [1, 4, 7, 10, 13, 16, 19, 22]

    residual_image_name = "267".zfill(6)
    concat_image_save_path = os.path.join(residual_root_path, f"concat_{residual_image_name}.png")

    for per_residual_id in residual_id:
        residual_np = np.load(os.path.join(residual_root_path, f"residual_images_{per_residual_id}/{residual_image_name}.npy"))
        zero_index = np.where(residual_np == 0)
        residual_image = cv2.imread(os.path.join(residual_root_path, f"visualization_{per_residual_id}/{residual_image_name}.png"))
        if per_residual_id == residual_id[0]:
            all_resdual_image = residual_image
        else:
            all_resdual_image = np.vstack((all_resdual_image, np.zeros((10, residual_image.shape[1], 3)), residual_image))

    cv2.imwrite(concat_image_save_path, all_resdual_image)