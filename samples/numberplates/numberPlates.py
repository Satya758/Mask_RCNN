import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


############################################################
#  Configurations
############################################################


class NPConfig(Config):
    # Give the configuration a recognizable name
    NAME = "np"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + balloon

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################


class NumberPlateDataset(utils.Dataset):

    def load_np(self, dataset_dir, subset):

        self.add_class("np", 1, "np")

        # assert subset in ["train", "val", "test"]

        img_dir = os.path.join(dataset_dir, subset)

        if subset not in ["samples"]:
            annotations = json.load(open(os.path.join(dataset_dir, "numberplates.json")))
            annotations = annotations["_via_img_metadata"]

            annotations = {k: v for k, v in annotations.items()
                           if v["regions"] and os.path.isfile(os.path.join(img_dir, v["filename"]))}

            for k, v in annotations.items():
                polygons = [r["shape_attributes"] for r in v["regions"]]
                filename = v["filename"]

                img_path = os.path.join(img_dir, filename)
                image = skimage.io.imread(img_path)
                height, width = image.shape[:2]

                self.add_image(
                    "np",
                    image_id=filename,
                    path=img_path,
                    width=width, height=height,
                    polygons=polygons)

                print(k)
        else:
            files = os.listdir(img_dir)

            annotations = json.load(open(os.path.join(dataset_dir, "numberplates.json")))
            annotations = annotations["_via_img_metadata"]

            annotations = {k: v for k, v in annotations.items()
                           if v["regions"] and os.path.isfile(os.path.join(img_dir, v["filename"]))}

            polygons = {}

            for k, v in annotations.items():
                polygons = [r["shape_attributes"] for r in v["regions"]]
                filenameD = v["filename"]

            for filename in files:
                img_path = os.path.join(img_dir, filename)

                image = skimage.io.imread(img_path)
                height, width = image.shape[:2]

                print(filename)

                self.add_image(
                    "np",
                    image_id=filename,
                    path=img_path,
                    width=width, height=height,
                    polygons=polygons)

    def load_mask(self, image_id):
        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        info = self.image_info[image_id]

        return info["path"]


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = NumberPlateDataset()
    dataset_train.load_np(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = NumberPlateDataset()
    dataset_val.load_np(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=30,
                layers='heads')


def color_splash(image, mask):
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255

    if mask.shape[-1] > 0:
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)

    return splash


def detect_and_color_splash(model, image_path):
    print("Running on {}".format(args.image))

    image = skimage.io.imread(args.image)

    r = model.detect([image], verbose=1)[0]

    splash = color_splash(image, r['masks'])

    file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
    skimage.io.imsave(file_name, splash)

    print("Saved to ", file_name)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect Number plates.')

    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")

    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/np/dataset/",
                        help='Directory of the Np dataset')

    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")

    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')

    args = parser.parse_args()

    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image, \
            "Provide --image to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", DEFAULT_LOGS_DIR)

    # Configurations
    if args.command == "train":
        config = NPConfig()
    else:
        class InferenceConfig(NPConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()

    print("Loading weights ", weights_path)

    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    print("Weights are loaded")
    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))
