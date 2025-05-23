import gcsfs
import tensorflow as tf
from .generic import *
import numpy as np
import io

class DataLoaderKittiRaw(DataLoaderGeneric):
    """Dataloader for the raw Kitti dataset from GCS using gcsfs
    """
    def __init__(self):
        super(DataLoaderKittiRaw, self).__init__('kitti-raw')

        self.in_size = [370, 1220]
        self.depth_type = "velodyne"

        # GCS filesystem
        self.fs = gcsfs.GCSFileSystem()

    def _set_output_size(self, out_size=[256, 768]):
        self.out_size = out_size
        crop = np.array([0.40810811 * out_size[0], 0.99189189 * out_size[0],
                         0.03594771 * out_size[1], 0.96405229 * out_size[1]]).astype(np.int32)
        crop_mask = np.zeros(self.out_size + [1])
        crop_mask[crop[0]:crop[1], crop[2]:crop[3], :] = 1
        self.eval_crop_mask = tf.convert_to_tensor(crop_mask, dtype=tf.float32)
    
    def _read_image_gcs(self, gcs_path):
        """Read image from GCS using gcsfs and return decoded JPEG tensor"""
        with self.fs.open(gcs_path, 'rb') as f:
            img_bytes = f.read()
        image = tf.io.decode_jpeg(img_bytes)  # or decode_png depending on use
        return image
    
    def _read_depth_gcs(self, gcs_path):
        """Read depth image from GCS using gcsfs and return decoded PNG tensor"""
        with self.fs.open(gcs_path, 'rb') as f:
            img_bytes = f.read()
        image = tf.image.decode_png(img_bytes, dtype=tf.uint16)
        return image

    #@tf.function
    def _decode_samples(self, data_sample):
        #image_path = tf.strings.join([self.db_path, data_sample['camera_l']], separator='/')
        image_path = f"{self.db_path}/{data_sample['camera_l'].numpy().decode()}"
        image = self._read_image_gcs(image_path)
        #file = tf.io.read_file(image_path)
        #file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['camera_l']], separator='/'))
        #image = tf.io.decode_jpeg(file)
        rgb_image = tf.cast(image, dtype=tf.float32)/255.

        camera_data = {
            "f": tf.convert_to_tensor([data_sample['fx']*self.out_size[1], data_sample['fy']*self.out_size[0]], dtype=tf.float32),
            "c": tf.convert_to_tensor([data_sample['cx']*self.out_size[1], data_sample['cy']*self.out_size[0]], dtype=tf.float32),
        }
        out_data = {}
        out_data["camera"] = camera_data.copy()
        out_data['RGB_im'] = tf.reshape(tf.image.resize(rgb_image, self.out_size), self.out_size+[3])
        out_data['rot'] = tf.cast(tf.stack([data_sample['qw'],data_sample['qx'],data_sample['qy'],data_sample['qz']], 0), dtype=tf.float32)
        out_data['trans'] = tf.cast(tf.stack([data_sample['tx'],data_sample['ty'],data_sample['tz']], 0), dtype=tf.float32)
        out_data['new_traj'] = tf.math.equal(data_sample['id'], 0)

        # Load depth data only if they are available
        if 'depth' in data_sample:
            #depth_path = tf.strings.join([self.db_path,"data_depth_annotated" ,data_sample['depth']], separator='/')
            depth_path = f"{self.db_path}/data_depth_annotated/{data_sample['depth'].numpy().decode()}"
            #file = tf.io.read_file(depth_path)
            depth = self._read_depth_gcs(depth_path)
            #file = tf.io.read_file(tf.strings.join([self.db_path, data_sample['depth']], separator='/'))
            #image = tf.image.decode_png(file, dtype=tf.uint16)
            depth = tf.cast(depth, dtype=tf.float32)/256
            out_data['depth'] = tf.reshape(tf.image.resize(depth, self.out_size, method='nearest'), self.out_size+[1])
            
            # crop used by Garg ECCV16 to reproduce Eigen NIPS14 results
            if self.usecase=="eval":
                out_data['depth'] = out_data['depth'] * self.eval_crop_mask

        return out_data

    def _perform_augmentation(self):
        #self._augmentation_step_flip()
        self._augmentation_step_color(invert_color=False)
