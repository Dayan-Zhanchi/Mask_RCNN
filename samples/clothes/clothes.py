"""
Mask R-CNN
Train on the deepfashion2 dataset.
"""

import os
import sys

import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils


############################################################
#  Configurations
############################################################

class ClothesConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "clothes"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100 / 2

    # Version of backbone network architecture
    # BACKBONE = "resnet50"

    # Non-maximum suppression threshold for detection
    # DETECTION_NMS_THRESHOLD = 0.5


class ClothesDataset(utils.Dataset):

    def load_clothes(self, number_of_data, dataset_dir, dataset_type, dataset_type_path):
        """Load a subset of the deepfashion2 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train, validation or test
        """
        self.add_class("clothes", 1, "short sleeve top")
        self.add_class("clothes", 2, "long sleeve top")
        self.add_class("clothes", 3, "short sleeve outwear")
        self.add_class("clothes", 4, "long sleeve outwear")
        self.add_class("clothes", 5, "vest")
        self.add_class("clothes", 6, "sling")
        self.add_class("clothes", 7, "shorts")
        self.add_class("clothes", 8, "trousers")
        self.add_class("clothes", 9, "skirt")
        self.add_class("clothes", 10, "short sleeve dress")
        self.add_class("clothes", 11, "long sleeve dress")
        self.add_class("clothes", 12, "vest dress")
        self.add_class("clothes", 13, "sling dress")

        # Train or validation dataset?
        assert dataset_type in ["train", "validation", "test"]
        dataset_dir = os.path.join(dataset_dir, dataset_type, dataset_type_path)

        coco = COCO(dataset_dir)
        # for idx, file in enumerate(os.walk(dataset_dir)[2]):
        #     if idx == number_of_data:
        #         break
        #     file_path = os.path.join(dataset_dir, file)
        #     annotations = json.load(open(file_path))
        #     # image_path = image_dir + str(idx).zfill(6) + '.jpg'
        #     # image = skimage.io.imread(image_path)
        #     # height, width = image.shape[:2]

        # Add images
        for i in range(1, number_of_data + 1):
            json_name = os.path.join(dataset_dir + "/" + str(i).zfill(6) + '.json')

            """we use coco.getAnnIds to get the corresponding annotations IDs for a given image ID. 
            Annotations are indexed at instance-level as opposed to image IDs which are indexed at image-level. 
            So number of image IDs < annotation IDs, since there can be more than 1 annotation in an image
            and therefore for a given image ID we can expect to get >= 1 annotations.
            """
            self.add_image("clothes",
                           image_id=i,  # we assume that the indices for the image_ids starts at 1 and keeps that format as they increment in value, i.e 1,2,3,4,5,.. etc
                           path=json_name,
                           width=coco.imgs[i]['width'],
                           height=coco.imgs[i]['height'],
                           annotations=coco.loadAnns(coco.getAnnIds([i], iscrowd=None)))

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "clothes":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a COCO image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "clothes":
            return super(self.__class__, self).load_mask(image_id)

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            # the class_id in this case will always correspond to the value of the category_id key in the given annotation
            class_id = self.map_source_class_id(
                "clothes.{}".format(annotation['category_id']))
            if class_id:
                m = self.annToMask(annotation, image_info[image_id]["images"]["height"],
                                   image_info[image_id]["images"]["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info[image_id]["images"]["height"] or m.shape[1] != image_info[image_id]["images"]["width"]:
                        m = np.ones([image_info[image_id]["images"]["height"], image_info[image_id]["images"]["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(self.__class__, self).load_mask(image_id)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann['segmentation']
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann['segmentation']
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

