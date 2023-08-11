import json
import argparse
import SimpleITK as sitk
import glob
from mmdet.apis import init_detector, inference_detector
from mmengine import Config
import math
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from albumentations.core.transforms_interface import ImageOnlyTransform
import torch.nn.functional as F
import numpy as np
from torch import nn
import timm # PyTorch Image Models
import torch
import copy
import time
import random
import os

list_ids = [
                          {"height": 1316, "width": 2892, "id": 1, "file_name": "val_15.png"},
                          {"height": 1316, "width": 2942, "id": 2, "file_name": "val_38.png"},
                          {"height": 1316, "width": 2987, "id": 3, "file_name": "val_33.png"},
                          {"height": 1504, "width": 2872, "id": 4, "file_name": "val_30.png"},
                          {"height": 1316, "width": 2970, "id": 5, "file_name": "val_5.png"},
                          {"height": 1316, "width": 2860, "id": 6, "file_name": "val_21.png"},
                          {"height": 1504, "width": 2804, "id": 7, "file_name": "val_39.png"},
                          {"height": 1316, "width": 2883, "id": 8, "file_name": "val_46.png"},
                          {"height": 1316, "width": 2967, "id": 9, "file_name": "val_20.png"},
                          {"height": 1504, "width": 2872, "id": 10, "file_name": "val_3.png"},
                          {"height": 1316, "width": 2954, "id": 11, "file_name": "val_29.png"},
                          {"height": 976, "width": 1976, "id": 12, "file_name": "val_2.png"},
                          {"height": 1316, "width": 2870, "id": 13, "file_name": "val_16.png"},
                          {"height": 1316, "width": 3004, "id": 14, "file_name": "val_25.png"},
                          {"height": 1316, "width": 2745, "id": 15, "file_name": "val_24.png"},
                          {"height": 1504, "width": 2872, "id": 16, "file_name": "val_31.png"},
                          {"height": 1316, "width": 2782, "id": 17, "file_name": "val_26.png"},
                          {"height": 1316, "width": 2744, "id": 18, "file_name": "val_44.png"},
                          {"height": 1504, "width": 2872, "id": 19, "file_name": "val_27.png"},
                          {"height": 1504, "width": 2868, "id": 20, "file_name": "val_41.png"},
                          {"height": 1316, "width": 3000, "id": 21, "file_name": "val_37.png"},
                          {"height": 1316, "width": 2797, "id": 22, "file_name": "val_40.png"},
                          {"height": 1316, "width": 2930, "id": 23, "file_name": "val_6.png"},
                          {"height": 1316, "width": 3003, "id": 24, "file_name": "val_18.png"},
                          {"height": 1316, "width": 2967, "id": 25, "file_name": "val_13.png"},
                          {"height": 1316, "width": 2822, "id": 26, "file_name": "val_8.png"},
                          {"height": 1316, "width": 2836, "id": 27, "file_name": "val_49.png"},
                          {"height": 1316, "width": 2704, "id": 28, "file_name": "val_23.png"},
                          {"height": 976, "width": 1976, "id": 29, "file_name": "val_1.png"},
                          {"height": 1504, "width": 2872, "id": 30, "file_name": "val_43.png"},
                          {"height": 1504, "width": 2872, "id": 31, "file_name": "val_28.png"},
                          {"height": 1504, "width": 2872, "id": 32, "file_name": "val_19.png"},
                          {"height": 1316, "width": 2728, "id": 33, "file_name": "val_14.png"},
                          {"height": 1316, "width": 2747, "id": 34, "file_name": "val_32.png"},
                          {"height": 976, "width": 1976, "id": 35, "file_name": "val_36.png"},
                          {"height": 1316, "width": 2829, "id": 36, "file_name": "val_47.png"},
                          {"height": 1316, "width": 2846, "id": 37, "file_name": "val_48.png"},
                          {"height": 1536, "width": 3076, "id": 38, "file_name": "val_17.png"},
                          {"height": 976, "width": 1976, "id": 39, "file_name": "val_42.png"},
                          {"height": 1504, "width": 2884, "id": 40, "file_name": "val_45.png"},
                          {"height": 1316, "width": 2741, "id": 41, "file_name": "val_9.png"},
                          {"height": 1316, "width": 2794, "id": 42, "file_name": "val_4.png"},
                          {"height": 1316, "width": 2959, "id": 43, "file_name": "val_34.png"},
                          {"height": 1316, "width": 2874, "id": 44, "file_name": "val_10.png"},
                          {"height": 1316, "width": 2978, "id": 45, "file_name": "val_35.png"},
                          {"height": 1504, "width": 2884, "id": 46, "file_name": "val_11.png"},
                          {"height": 1316, "width": 2794, "id": 47, "file_name": "val_12.png"},
                          {"height": 1316, "width": 2959, "id": 48, "file_name": "val_7.png"},
                          {"height": 1316, "width": 2912, "id": 49, "file_name": "val_22.png"},
                          {"height": 1504, "width": 2872, "id": 50, "file_name": "val_0.png"},
                      ]

def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help="Minimum score for instance predictions to be shown",
    )

    parser.add_argument(
        "--nclass",
        type=int,
        default=3,
        help="Number of trained classes",
    )

    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


class Loginverse(ImageOnlyTransform):
    def apply(self, img, **params):
      normalized_image = img / 255.0

      # Set the gamma value for inverse log transform
      gamma = 1.5 # Adjust the gamma value as needed

      # Apply inverse log transform (power-law transformation)
      inverse_log_transformed_image = np.power(normalized_image, gamma)

      # Rescale the image back to [0, 255]
      filtered_image = (inverse_log_transformed_image * 255).astype(np.uint8)
      return filtered_image

class Transforms:
    def __init__(self, transforms: A.Compose):
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        return self.transforms(image=np.array(img))


class Hierarchialdet:
    def __init__(self):
        self.cfg = None
        self.demo = None
        self.input_dir = "input"

    def setup(self):
        args = get_parser().parse_args()
        self.seed_everything(0)
        device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else 'cpu')
        self.device = device
        print(device)
        print("init enum model..")
        config_file = '/opt/app/configs/swintest.py'
        cfg = Config.fromfile(config_file)
        checkpoint_file = '/opt/app/configs/epoch_12.pth'
        self.enum_model = init_detector(cfg, checkpoint_file, device=device)

        print("init model..")
        config_file1 = '/opt/app/configs/dinodisease.py'
        cfg1 = Config.fromfile(config_file1)
        checkpoint_file1 = '/opt/app/configs/epoch_51.pth'
        self.model = init_detector(cfg1, checkpoint_file1, device=device)


        # config_file2 = '/opt/app/configs/dinoswin.py'
        # cfg2 = Config.fromfile(config_file2)
        # checkpoint_file2 = '/opt/app/configs/superbest14800.pth'
        # self.modeldiff = init_detector(cfg2, checkpoint_file2, device=device)

        self.Threshold_enum = 0.7
        self.Threshold = 0.05
        self.Thresholddino = 0.05

        self.CLASSES = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28',
                        '31',
                        '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48']
        self.cat = ['Caries', 'Deep Caries', 'Impacted', 'Periapical Lesion']
        self.cattoid = {'Caries': 1, 'Deep Caries': 3, 'Impacted': 0, 'Periapical Lesion': 2}
        ##two step###
        # self.test_transform = A.Compose([
        #     A.LongestMaxSize(max_size=224, interpolation=1),
        #     A.PadIfNeeded(min_height=224, min_width=224, border_mode=0, value=(0, 0, 0)),
        #     Loginverse(always_apply=True),
        #
        #     A.Normalize(mean=0.5, std=0.2, max_pixel_value=255.0, always_apply=True, p=1.0),
        #     ToTensorV2()
        # ])
        # self.caries = self.create_model('/opt/app/configs/carieslog.pt').to(device)
        # #self.deepcaries = self.create_model('/opt/app/configs/deepcarieslog1.pt').to(device)
        # self.impacted = self.create_model('/opt/app/configs/impactedlog.pt').to(device)
        # self.peri = self.create_model('/opt/app/configs/pallog.pt').to(device)

    def process(self):
        self.setup()

        file_path = glob.glob('/input/test/*.mha')[0]
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        print("test..")
        print(image_array.shape)
        detection = {
            "name": "Regions of interest",
            "type": "Multiple 2D bounding boxes",
            "boxes": [],
            "version": {"major": 1, "minor": 0}}
        boxes = []
        start_time = time.time()
        print('start:', start_time)
        for k in range(image_array.shape[2]):
            image_name = "val_{}.png".format(k)
            for input_img in list_ids:
                if input_img["file_name"] == image_name:
                    img_id = input_img["id"]
            print(img_id)
            k_boxes = self.run_on_image(image_array[:, :, k, :], img_id)
            boxes += k_boxes
            if k == 1:
                break
        print('end', time.time()-start_time)

        detection["boxes"] = boxes

        output_file = "/output/abnormal-teeth-detection.json"
        with open(output_file, "w") as f:
            json.dump(detection, f)

        print("Inference completed. Results saved to", output_file)

    def run_on_image(self, img, img_id):
        enumeration = {}
        enumerationscore = {}

        pred_enum = inference_detector(self.enum_model, img)
        pred = pred_enum.pred_instances.cpu().numpy()
        boxes = []
        for i, score in enumerate(pred.scores[pred.scores > self.Threshold_enum]):
            enum = pred.labels[i]
            if enum - 1 < 32:
                enum = self.CLASSES[enum - 1]
            else:
                continue

            enum = int(enum)
            bbox = pred.bboxes[i]
            x_ref, y_ref = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
            enumeration[str(enum)] = (x_ref, y_ref)
            enumerationscore[str(enum)] = score

            new_result = inference_detector(self.model, img)
            pred = new_result.pred_instances.cpu().numpy()
            boxes = []
            for i, score in enumerate(pred.scores[pred.scores > self.Threshold]):
                output = {}
                bbox = pred.bboxes[i]
                x, y = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
                num = self.find_closest_keys(enumeration, (x, y))

                disease = pred.labels[i]

                cat1 = int(num / 10) - 1
                cat2 = num % 10 - 1
                cat3 = self.cattoid[self.cat[disease - 1]]

                corners = [[float(bbox[0]), float(bbox[1]), img_id], [float(bbox[0]), float(bbox[3]), img_id],
                           [float(bbox[2]), float(bbox[1]), img_id], [float(bbox[2]), float(bbox[3]), img_id]]
                # [x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
                output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
                output['corners'] = corners
                output['probability'] = float(score * enumerationscore[str(num)])
                boxes.append(output)

            new_result = inference_detector(self.model, img)
            pred = new_result.pred_instances.cpu().numpy()

            for i, score in enumerate(pred.scores[pred.scores < self.Thresholddino]):
                output = {}
                bbox = pred.bboxes[i]
                x, y = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
                num = self.find_closest_keys(enumeration, (x, y))

                disease = pred.labels[i]

                cat1 = int(num / 10) - 1
                cat2 = num % 10 - 1
                cat3 = self.cattoid[self.cat[disease - 1]]

                corners = [[float(bbox[0]), float(bbox[1]), img_id], [float(bbox[0]), float(bbox[3]), img_id],
                           [float(bbox[2]), float(bbox[1]), img_id], [float(bbox[2]), float(bbox[3]), img_id]]
                # [x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
                output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
                output['corners'] = corners
                output['probability'] = float(score * enumerationscore[str(num)])
                boxes.append(output)

        return boxes

    def find_closest_keys(self, dictionary, reference_value):
        closest_keys = []
        min_distance = float('inf')

        for key, value in dictionary.items():
            distance = math.sqrt((value[0] - reference_value[0]) ** 2 + (value[1] - reference_value[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                closest_keys = [key]
            elif distance == min_distance:
                closest_keys.append(key)

        return int(closest_keys[0])

    def create_model(self, dir):
        cariesmodel = timm.create_model('tf_efficientnet_b4_ns',pretrained=False) #load pretrained model
        #let's update the pretarined model:
        for param in cariesmodel.parameters():
            param.requires_grad=True

        cariesmodel.classifier = nn.Sequential(
          nn.Linear(in_features=1792, out_features=2)
        )
        cariesmodel.load_state_dict(torch.load(dir, map_location=self.device))
        cariesmodel.eval()

        return cariesmodel

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    Hierarchialdet().process()
