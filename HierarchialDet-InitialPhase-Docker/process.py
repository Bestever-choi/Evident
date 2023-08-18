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
             {
                 "height":1316,
                 "width":2810,
                 "id":1,
                 "file_name":"test_42.png"
              },
              {
                 "height":1316,
                 "width":2985,
                 "id":2,
                 "file_name":"test_171.png"
              },
              {
                 "height":1316,
                 "width":2851,
                 "id":3,
                 "file_name":"test_229.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":4,
                 "file_name":"test_26.png"
              },
              {
                 "height":1316,
                 "width":2904,
                 "id":5,
                 "file_name":"test_73.png"
              },
              {
                 "height":1316,
                 "width":2831,
                 "id":6,
                 "file_name":"test_63.png"
              },
              {
                 "height":1316,
                 "width":2799,
                 "id":7,
                 "file_name":"test_138.png"
              },
              {
                 "height":1504,
                 "width":2680,
                 "id":8,
                 "file_name":"test_147.png"
              },
              {
                 "height":1316,
                 "width":2873,
                 "id":9,
                 "file_name":"test_79.png"
              },
              {
                 "height":1316,
                 "width":2847,
                 "id":10,
                 "file_name":"test_144.png"
              },
              {
                 "height":1316,
                 "width":2808,
                 "id":11,
                 "file_name":"test_31.png"
              },
              {
                 "height":1316,
                 "width":2969,
                 "id":12,
                 "file_name":"test_127.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":13,
                 "file_name":"test_143.png"
              },
              {
                 "height":1504,
                 "width":2580,
                 "id":14,
                 "file_name":"test_43.png"
              },
              {
                 "height":1316,
                 "width":2915,
                 "id":15,
                 "file_name":"test_101.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":16,
                 "file_name":"test_201.png"
              },
              {
                 "height":1316,
                 "width":2895,
                 "id":17,
                 "file_name":"test_176.png"
              },
              {
                 "height":1316,
                 "width":2836,
                 "id":18,
                 "file_name":"test_222.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":19,
                 "file_name":"test_9.png"
              },
              {
                 "height":1316,
                 "width":2967,
                 "id":20,
                 "file_name":"test_61.png"
              },
              {
                 "height":1316,
                 "width":2749,
                 "id":21,
                 "file_name":"test_52.png"
              },
              {
                 "height":1316,
                 "width":2836,
                 "id":22,
                 "file_name":"test_245.png"
              },
              {
                 "height":1316,
                 "width":2892,
                 "id":23,
                 "file_name":"test_208.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":24,
                 "file_name":"test_247.png"
              },
              {
                 "height":1316,
                 "width":2991,
                 "id":25,
                 "file_name":"test_47.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":26,
                 "file_name":"test_95.png"
              },
              {
                 "height":1316,
                 "width":2787,
                 "id":27,
                 "file_name":"test_34.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":28,
                 "file_name":"test_178.png"
              },
              {
                 "height":1316,
                 "width":2950,
                 "id":29,
                 "file_name":"test_218.png"
              },
              {
                 "height":1316,
                 "width":2654,
                 "id":30,
                 "file_name":"test_68.png"
              },
              {
                 "height":1316,
                 "width":2935,
                 "id":31,
                 "file_name":"test_100.png"
              },
              {
                 "height":1504,
                 "width":2884,
                 "id":32,
                 "file_name":"test_105.png"
              },
              {
                 "height":1316,
                 "width":2949,
                 "id":33,
                 "file_name":"test_149.png"
              },
              {
                 "height":1504,
                 "width":2876,
                 "id":34,
                 "file_name":"test_228.png"
              },
              {
                 "height":1316,
                 "width":2878,
                 "id":35,
                 "file_name":"test_134.png"
              },
              {
                 "height":1504,
                 "width":2516,
                 "id":36,
                 "file_name":"test_30.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":37,
                 "file_name":"test_84.png"
              },
              {
                 "height":1316,
                 "width":2931,
                 "id":38,
                 "file_name":"test_2.png"
              },
              {
                 "height":1316,
                 "width":2948,
                 "id":39,
                 "file_name":"test_244.png"
              },
              {
                 "height":1316,
                 "width":2983,
                 "id":40,
                 "file_name":"test_181.png"
              },
              {
                 "height":1316,
                 "width":2870,
                 "id":41,
                 "file_name":"test_200.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":42,
                 "file_name":"test_90.png"
              },
              {
                 "height":1316,
                 "width":2944,
                 "id":43,
                 "file_name":"test_112.png"
              },
              {
                 "height":1316,
                 "width":2880,
                 "id":44,
                 "file_name":"test_238.png"
              },
              {
                 "height":1316,
                 "width":2892,
                 "id":45,
                 "file_name":"test_231.png"
              },
              {
                 "height":1316,
                 "width":2844,
                 "id":46,
                 "file_name":"test_13.png"
              },
              {
                 "height":1504,
                 "width":2516,
                 "id":47,
                 "file_name":"test_240.png"
              },
              {
                 "height":1316,
                 "width":2792,
                 "id":48,
                 "file_name":"test_32.png"
              },
              {
                 "height":1316,
                 "width":2809,
                 "id":49,
                 "file_name":"test_133.png"
              },
              {
                 "height":1316,
                 "width":2958,
                 "id":50,
                 "file_name":"test_77.png"
              },
              {
                 "height":1316,
                 "width":2844,
                 "id":51,
                 "file_name":"test_166.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":52,
                 "file_name":"test_177.png"
              },
              {
                 "height":1316,
                 "width":2834,
                 "id":53,
                 "file_name":"test_183.png"
              },
              {
                 "height":1316,
                 "width":2936,
                 "id":54,
                 "file_name":"test_83.png"
              },
              {
                 "height":1316,
                 "width":2854,
                 "id":55,
                 "file_name":"test_81.png"
              },
              {
                 "height":1316,
                 "width":2765,
                 "id":56,
                 "file_name":"test_93.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":57,
                 "file_name":"test_5.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":58,
                 "file_name":"test_60.png"
              },
              {
                 "height":1316,
                 "width":2966,
                 "id":59,
                 "file_name":"test_207.png"
              },
              {
                 "height":1316,
                 "width":2941,
                 "id":60,
                 "file_name":"test_194.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":61,
                 "file_name":"test_114.png"
              },
              {
                 "height":1316,
                 "width":2812,
                 "id":62,
                 "file_name":"test_71.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":63,
                 "file_name":"test_132.png"
              },
              {
                 "height":1316,
                 "width":2952,
                 "id":64,
                 "file_name":"test_209.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":65,
                 "file_name":"test_148.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":66,
                 "file_name":"test_102.png"
              },
              {
                 "height":1316,
                 "width":2953,
                 "id":67,
                 "file_name":"test_56.png"
              },
              {
                 "height":1316,
                 "width":2904,
                 "id":68,
                 "file_name":"test_20.png"
              },
              {
                 "height":1316,
                 "width":2940,
                 "id":69,
                 "file_name":"test_10.png"
              },
              {
                 "height":1316,
                 "width":2801,
                 "id":70,
                 "file_name":"test_151.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":71,
                 "file_name":"test_54.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":72,
                 "file_name":"test_236.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":73,
                 "file_name":"test_216.png"
              },
              {
                 "height":1316,
                 "width":2856,
                 "id":74,
                 "file_name":"test_227.png"
              },
              {
                 "height":1316,
                 "width":2721,
                 "id":75,
                 "file_name":"test_12.png"
              },
              {
                 "height":1316,
                 "width":2807,
                 "id":76,
                 "file_name":"test_164.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":77,
                 "file_name":"test_120.png"
              },
              {
                 "height":1316,
                 "width":2807,
                 "id":78,
                 "file_name":"test_28.png"
              },
              {
                 "height":1316,
                 "width":2918,
                 "id":79,
                 "file_name":"test_142.png"
              },
              {
                 "height":1316,
                 "width":3000,
                 "id":80,
                 "file_name":"test_109.png"
              },
              {
                 "height":1316,
                 "width":2702,
                 "id":81,
                 "file_name":"test_155.png"
              },
              {
                 "height":1316,
                 "width":3001,
                 "id":82,
                 "file_name":"test_82.png"
              },
              {
                 "height":1504,
                 "width":2880,
                 "id":83,
                 "file_name":"test_198.png"
              },
              {
                 "height":1316,
                 "width":2940,
                 "id":84,
                 "file_name":"test_98.png"
              },
              {
                 "height":1316,
                 "width":2820,
                 "id":85,
                 "file_name":"test_168.png"
              },
              {
                 "height":1316,
                 "width":2872,
                 "id":86,
                 "file_name":"test_91.png"
              },
              {
                 "height":1316,
                 "width":2891,
                 "id":87,
                 "file_name":"test_128.png"
              },
              {
                 "height":1316,
                 "width":2664,
                 "id":88,
                 "file_name":"test_22.png"
              },
              {
                 "height":1504,
                 "width":2884,
                 "id":89,
                 "file_name":"test_160.png"
              },
              {
                 "height":1316,
                 "width":2867,
                 "id":90,
                 "file_name":"test_19.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":91,
                 "file_name":"test_89.png"
              },
              {
                 "height":1316,
                 "width":2828,
                 "id":92,
                 "file_name":"test_85.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":93,
                 "file_name":"test_180.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":94,
                 "file_name":"test_18.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":95,
                 "file_name":"test_175.png"
              },
              {
                 "height":1316,
                 "width":2891,
                 "id":96,
                 "file_name":"test_24.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":97,
                 "file_name":"test_97.png"
              },
              {
                 "height":1316,
                 "width":2916,
                 "id":98,
                 "file_name":"test_117.png"
              },
              {
                 "height":1504,
                 "width":2884,
                 "id":99,
                 "file_name":"test_126.png"
              },
              {
                 "height":1316,
                 "width":2832,
                 "id":100,
                 "file_name":"test_234.png"
              },
              {
                 "height":1316,
                 "width":2926,
                 "id":101,
                 "file_name":"test_221.png"
              },
              {
                 "height":1316,
                 "width":2893,
                 "id":102,
                 "file_name":"test_115.png"
              },
              {
                 "height":1316,
                 "width":2848,
                 "id":103,
                 "file_name":"test_161.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":104,
                 "file_name":"test_223.png"
              },
              {
                 "height":1316,
                 "width":2829,
                 "id":105,
                 "file_name":"test_0.png"
              },
              {
                 "height":1316,
                 "width":2988,
                 "id":106,
                 "file_name":"test_173.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":107,
                 "file_name":"test_104.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":108,
                 "file_name":"test_235.png"
              },
              {
                 "height":1316,
                 "width":2874,
                 "id":109,
                 "file_name":"test_196.png"
              },
              {
                 "height":1316,
                 "width":2928,
                 "id":110,
                 "file_name":"test_87.png"
              },
              {
                 "height":1316,
                 "width":2771,
                 "id":111,
                 "file_name":"test_107.png"
              },
              {
                 "height":1316,
                 "width":2927,
                 "id":112,
                 "file_name":"test_72.png"
              },
              {
                 "height":1316,
                 "width":2708,
                 "id":113,
                 "file_name":"test_249.png"
              },
              {
                 "height":1316,
                 "width":2918,
                 "id":114,
                 "file_name":"test_94.png"
              },
              {
                 "height":1316,
                 "width":2963,
                 "id":115,
                 "file_name":"test_33.png"
              },
              {
                 "height":1316,
                 "width":2955,
                 "id":116,
                 "file_name":"test_110.png"
              },
              {
                 "height":1316,
                 "width":2836,
                 "id":117,
                 "file_name":"test_191.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":118,
                 "file_name":"test_193.png"
              },
              {
                 "height":1316,
                 "width":2945,
                 "id":119,
                 "file_name":"test_121.png"
              },
              {
                 "height":1316,
                 "width":2887,
                 "id":120,
                 "file_name":"test_237.png"
              },
              {
                 "height":1316,
                 "width":2968,
                 "id":121,
                 "file_name":"test_157.png"
              },
              {
                 "height":1316,
                 "width":2889,
                 "id":122,
                 "file_name":"test_55.png"
              },
              {
                 "height":1316,
                 "width":2956,
                 "id":123,
                 "file_name":"test_130.png"
              },
              {
                 "height":1316,
                 "width":2843,
                 "id":124,
                 "file_name":"test_108.png"
              },
              {
                 "height":1316,
                 "width":2834,
                 "id":125,
                 "file_name":"test_3.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":126,
                 "file_name":"test_217.png"
              },
              {
                 "height":1316,
                 "width":2932,
                 "id":127,
                 "file_name":"test_210.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":128,
                 "file_name":"test_65.png"
              },
              {
                 "height":1504,
                 "width":2508,
                 "id":129,
                 "file_name":"test_158.png"
              },
              {
                 "height":1316,
                 "width":2752,
                 "id":130,
                 "file_name":"test_145.png"
              },
              {
                 "height":1316,
                 "width":2830,
                 "id":131,
                 "file_name":"test_169.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":132,
                 "file_name":"test_152.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":133,
                 "file_name":"test_41.png"
              },
              {
                 "height":1316,
                 "width":2856,
                 "id":134,
                 "file_name":"test_248.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":135,
                 "file_name":"test_59.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":136,
                 "file_name":"test_96.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":137,
                 "file_name":"test_136.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":138,
                 "file_name":"test_179.png"
              },
              {
                 "height":1316,
                 "width":2916,
                 "id":139,
                 "file_name":"test_241.png"
              },
              {
                 "height":1316,
                 "width":2885,
                 "id":140,
                 "file_name":"test_4.png"
              },
              {
                 "height":1316,
                 "width":2921,
                 "id":141,
                 "file_name":"test_80.png"
              },
              {
                 "height":1316,
                 "width":2776,
                 "id":142,
                 "file_name":"test_119.png"
              },
              {
                 "height":1504,
                 "width":2492,
                 "id":143,
                 "file_name":"test_226.png"
              },
              {
                 "height":1316,
                 "width":2965,
                 "id":144,
                 "file_name":"test_88.png"
              },
              {
                 "height":1316,
                 "width":2762,
                 "id":145,
                 "file_name":"test_204.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":146,
                 "file_name":"test_15.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":147,
                 "file_name":"test_51.png"
              },
              {
                 "height":1316,
                 "width":2954,
                 "id":148,
                 "file_name":"test_44.png"
              },
              {
                 "height":1316,
                 "width":2853,
                 "id":149,
                 "file_name":"test_29.png"
              },
              {
                 "height":1316,
                 "width":2902,
                 "id":150,
                 "file_name":"test_139.png"
              },
              {
                 "height":1316,
                 "width":2896,
                 "id":151,
                 "file_name":"test_45.png"
              },
              {
                 "height":1316,
                 "width":2951,
                 "id":152,
                 "file_name":"test_170.png"
              },
              {
                 "height":1504,
                 "width":2648,
                 "id":153,
                 "file_name":"test_75.png"
              },
              {
                 "height":1316,
                 "width":2865,
                 "id":154,
                 "file_name":"test_137.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":155,
                 "file_name":"test_242.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":156,
                 "file_name":"test_188.png"
              },
              {
                 "height":1316,
                 "width":2817,
                 "id":157,
                 "file_name":"test_99.png"
              },
              {
                 "height":1316,
                 "width":2842,
                 "id":158,
                 "file_name":"test_78.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":159,
                 "file_name":"test_38.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":160,
                 "file_name":"test_21.png"
              },
              {
                 "height":1504,
                 "width":2884,
                 "id":161,
                 "file_name":"test_224.png"
              },
              {
                 "height":1316,
                 "width":2874,
                 "id":162,
                 "file_name":"test_214.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":163,
                 "file_name":"test_124.png"
              },
              {
                 "height":1316,
                 "width":2911,
                 "id":164,
                 "file_name":"test_67.png"
              },
              {
                 "height":1316,
                 "width":2862,
                 "id":165,
                 "file_name":"test_185.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":166,
                 "file_name":"test_16.png"
              },
              {
                 "height":1316,
                 "width":2935,
                 "id":167,
                 "file_name":"test_14.png"
              },
              {
                 "height":1316,
                 "width":2832,
                 "id":168,
                 "file_name":"test_167.png"
              },
              {
                 "height":1316,
                 "width":2878,
                 "id":169,
                 "file_name":"test_146.png"
              },
              {
                 "height":1316,
                 "width":2896,
                 "id":170,
                 "file_name":"test_232.png"
              },
              {
                 "height":1316,
                 "width":2896,
                 "id":171,
                 "file_name":"test_215.png"
              },
              {
                 "height":1316,
                 "width":2750,
                 "id":172,
                 "file_name":"test_165.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":173,
                 "file_name":"test_6.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":174,
                 "file_name":"test_233.png"
              },
              {
                 "height":1316,
                 "width":2846,
                 "id":175,
                 "file_name":"test_225.png"
              },
              {
                 "height":1504,
                 "width":2892,
                 "id":176,
                 "file_name":"test_220.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":177,
                 "file_name":"test_172.png"
              },
              {
                 "height":1316,
                 "width":2954,
                 "id":178,
                 "file_name":"test_163.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":179,
                 "file_name":"test_62.png"
              },
              {
                 "height":1316,
                 "width":2827,
                 "id":180,
                 "file_name":"test_213.png"
              },
              {
                 "height":1316,
                 "width":2886,
                 "id":181,
                 "file_name":"test_243.png"
              },
              {
                 "height":1316,
                 "width":2953,
                 "id":182,
                 "file_name":"test_53.png"
              },
              {
                 "height":1316,
                 "width":2808,
                 "id":183,
                 "file_name":"test_66.png"
              },
              {
                 "height":1316,
                 "width":2840,
                 "id":184,
                 "file_name":"test_190.png"
              },
              {
                 "height":1504,
                 "width":2636,
                 "id":185,
                 "file_name":"test_182.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":186,
                 "file_name":"test_154.png"
              },
              {
                 "height":1316,
                 "width":2946,
                 "id":187,
                 "file_name":"test_64.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":188,
                 "file_name":"test_156.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":189,
                 "file_name":"test_35.png"
              },
              {
                 "height":1316,
                 "width":2812,
                 "id":190,
                 "file_name":"test_206.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":191,
                 "file_name":"test_186.png"
              },
              {
                 "height":1316,
                 "width":2798,
                 "id":192,
                 "file_name":"test_131.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":193,
                 "file_name":"test_76.png"
              },
              {
                 "height":1316,
                 "width":2832,
                 "id":194,
                 "file_name":"test_199.png"
              },
              {
                 "height":1316,
                 "width":2957,
                 "id":195,
                 "file_name":"test_129.png"
              },
              {
                 "height":1316,
                 "width":2798,
                 "id":196,
                 "file_name":"test_7.png"
              },
              {
                 "height":1316,
                 "width":2902,
                 "id":197,
                 "file_name":"test_11.png"
              },
              {
                 "height":1316,
                 "width":2809,
                 "id":198,
                 "file_name":"test_150.png"
              },
              {
                 "height":1316,
                 "width":2727,
                 "id":199,
                 "file_name":"test_40.png"
              },
              {
                 "height":1316,
                 "width":2805,
                 "id":200,
                 "file_name":"test_184.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":201,
                 "file_name":"test_135.png"
              },
              {
                 "height":1316,
                 "width":2836,
                 "id":202,
                 "file_name":"test_23.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":203,
                 "file_name":"test_69.png"
              },
              {
                 "height":1316,
                 "width":2877,
                 "id":204,
                 "file_name":"test_25.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":205,
                 "file_name":"test_1.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":206,
                 "file_name":"test_122.png"
              },
              {
                 "height":1316,
                 "width":2864,
                 "id":207,
                 "file_name":"test_153.png"
              },
              {
                 "height":1316,
                 "width":2967,
                 "id":208,
                 "file_name":"test_212.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":209,
                 "file_name":"test_36.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":210,
                 "file_name":"test_140.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":211,
                 "file_name":"test_239.png"
              },
              {
                 "height":1316,
                 "width":2812,
                 "id":212,
                 "file_name":"test_111.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":213,
                 "file_name":"test_197.png"
              },
              {
                 "height":1316,
                 "width":2878,
                 "id":214,
                 "file_name":"test_37.png"
              },
              {
                 "height":1316,
                 "width":2772,
                 "id":215,
                 "file_name":"test_57.png"
              },
              {
                 "height":1316,
                 "width":2864,
                 "id":216,
                 "file_name":"test_113.png"
              },
              {
                 "height":1316,
                 "width":2985,
                 "id":217,
                 "file_name":"test_58.png"
              },
              {
                 "height":1316,
                 "width":2923,
                 "id":218,
                 "file_name":"test_187.png"
              },
              {
                 "height":1504,
                 "width":2544,
                 "id":219,
                 "file_name":"test_230.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":220,
                 "file_name":"test_174.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":221,
                 "file_name":"test_8.png"
              },
              {
                 "height":1504,
                 "width":2888,
                 "id":222,
                 "file_name":"test_192.png"
              },
              {
                 "height":1316,
                 "width":2736,
                 "id":223,
                 "file_name":"test_202.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":224,
                 "file_name":"test_125.png"
              },
              {
                 "height":1316,
                 "width":2950,
                 "id":225,
                 "file_name":"test_205.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":226,
                 "file_name":"test_103.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":227,
                 "file_name":"test_195.png"
              },
              {
                 "height":1316,
                 "width":2880,
                 "id":228,
                 "file_name":"test_49.png"
              },
              {
                 "height":1316,
                 "width":2859,
                 "id":229,
                 "file_name":"test_74.png"
              },
              {
                 "height":1316,
                 "width":2970,
                 "id":230,
                 "file_name":"test_106.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":231,
                 "file_name":"test_123.png"
              },
              {
                 "height":1316,
                 "width":2878,
                 "id":232,
                 "file_name":"test_17.png"
              },
              {
                 "height":1316,
                 "width":2948,
                 "id":233,
                 "file_name":"test_39.png"
              },
              {
                 "height":1316,
                 "width":2780,
                 "id":234,
                 "file_name":"test_50.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":235,
                 "file_name":"test_246.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":236,
                 "file_name":"test_211.png"
              },
              {
                 "height":1316,
                 "width":2773,
                 "id":237,
                 "file_name":"test_46.png"
              },
              {
                 "height":1316,
                 "width":2901,
                 "id":238,
                 "file_name":"test_70.png"
              },
              {
                 "height":1536,
                 "width":3076,
                 "id":239,
                 "file_name":"test_27.png"
              },
              {
                 "height":1316,
                 "width":2903,
                 "id":240,
                 "file_name":"test_141.png"
              },
              {
                 "height":1504,
                 "width":2868,
                 "id":241,
                 "file_name":"test_162.png"
              },
              {
                 "height":1316,
                 "width":2954,
                 "id":242,
                 "file_name":"test_189.png"
              },
              {
                 "height":976,
                 "width":1976,
                 "id":243,
                 "file_name":"test_86.png"
              },
              {
                 "height":1504,
                 "width":2872,
                 "id":244,
                 "file_name":"test_219.png"
              },
              {
                 "height":1316,
                 "width":2824,
                 "id":245,
                 "file_name":"test_159.png"
              },
              {
                 "height":1316,
                 "width":2808,
                 "id":246,
                 "file_name":"test_203.png"
              },
              {
                 "height":1316,
                 "width":2875,
                 "id":247,
                 "file_name":"test_116.png"
              },
              {
                 "height":1316,
                 "width":3002,
                 "id":248,
                 "file_name":"test_48.png"
              },
              {
                 "height":1316,
                 "width":2866,
                 "id":249,
                 "file_name":"test_118.png"
              },
              {
                 "height":1316,
                 "width":2796,
                 "id":250,
                 "file_name":"test_92.png"
              }
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
        checkpoint_file1 = '/opt/app/configs/epoch_57.pth'
        self.model = init_detector(cfg1, checkpoint_file1, device=device)

        config_file2 = '/opt/app/configs/dinoswin.py'
        cfg2 = Config.fromfile(config_file2)
        checkpoint_file2 = '/opt/app/configs/superbest15400.pth'
        self.modeldiff = init_detector(cfg2, checkpoint_file2, device=device)

        self.Threshold_enum = 0.7
        self.Threshold = 0.01
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
        
        file_path = glob.glob('/input/images/panoramic-dental-xrays/*.mha')[0]
        image = sitk.ReadImage(file_path)
        image_array = sitk.GetArrayFromImage(image)
        print("test..")

        detection = {
            "name": "Regions of interest",
            "type": "Multiple 2D bounding boxes",
            "boxes": [],
            "version": {"major": 1, "minor": 0}}
        boxes = []
        start_time = time.time()
        print('start:', start_time)
        for k in range(image_array.shape[2]):
            image_name = "test_{}.png".format(k)
            for input_img in list_ids:
                if input_img["file_name"] == image_name:
                    img_id = input_img["id"]
            print(img_id)
            k_boxes = self.run_on_image(image_array[:, :, k, :], img_id)
            boxes += k_boxes
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

            # output = {}
            # cat1 = int(enum / 10) - 1
            # cat2 = enum % 10 - 1
            # corners = [[float(bbox[0]), float(bbox[1]), img_id], [float(bbox[0]), float(bbox[3]), img_id],
            #            [float(bbox[2]), float(bbox[1]), img_id], [float(bbox[2]), float(bbox[3]), img_id]]
            #
            # bbox_r = [round(i) for i in bbox]
            # crop1 = img[bbox_r[1]:bbox_r[3], bbox_r[0]:bbox_r[2], :]
            # crop = self.test_transform(image=crop1)["image"]
            #
            # output['corners'] = corners
            #
            # if enum % 10 == 8:
            #     psi = self.impacted(crop.to(self.device).unsqueeze(0))
            #     psi = F.softmax(psi, dim=1)
            #     if psi[0][0] > 0.7:
            #         cat3 = 0
            #         s = psi[0][0]
            #         output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
            #         output['probability'] = float(score * s)
            #         boxes.append(copy.deepcopy(output))
            # else:
            #     ps = self.caries(crop.to(self.device).unsqueeze(0))
            #     ps = F.softmax(ps, dim=1)
            #     if ps[0][0] > 0.5:
            #         cat3 = 1
            #         s = ps[0][0]
            #         output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
            #         output['probability'] = float(score * s)
            #         boxes.append(copy.deepcopy(output))
            #
            #     psp = self.peri(crop.to(self.device).unsqueeze(0))
            #     psp = F.softmax(psp, dim=1)
            #     if psp[0][1] > 0.5:
            #         cat3 = 2
            #         s = psp[0][1]
            #         output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
            #         output['probability'] = float(score * s)
            #         boxes.append(copy.deepcopy(output))

        new_result_diff = inference_detector(self.modeldiff, img)
        pred_diff = new_result_diff.pred_instances.cpu().numpy()
        for i, score in enumerate(pred_diff.scores[pred_diff.scores > self.Threshold]):
            output = {}
            bbox = pred_diff.bboxes[i]
            x, y = ((bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2)
            num = self.find_closest_keys(enumeration, (x, y))

            disease = pred_diff.labels[i]

            cat1 = int(num / 10) - 1
            cat2 = num % 10 - 1
            cat3 = self.cattoid[self.cat[disease - 1]]

            corners = [[float(bbox[0]), float(bbox[1]), img_id], [float(bbox[0]), float(bbox[3]), img_id],
                       [float(bbox[2]), float(bbox[1]), img_id], [float(bbox[2]), float(bbox[3]), img_id]]
            # [x1, y1, image_id], [x2, y2, image_id], [x3, y3, image_id], [x4, y4, image_id]
            output['name'] = str(cat1) + '-' + str(cat2) + '-' + str(cat3)
            output['corners'] = corners
            output['probability'] = float(score * enumerationscore[str(num)])
            boxes.append(copy.deepcopy(output))

        new_result = inference_detector(self.model, img)
        pred = new_result.pred_instances.cpu().numpy()
        for i, score in enumerate(pred.scores[pred.scores < self.Threshold]):
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
            boxes.append(copy.deepcopy(output))

        return boxes

    def find_closest_keys(self, dictionary, reference_value):
        closest_keys = []
        min_distance = float('inf')

        for key, value in dictionary.items():
            distance = math.sqrt((value[0] - reference_value[0])**2 + (value[1] - reference_value[1])**2)
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
