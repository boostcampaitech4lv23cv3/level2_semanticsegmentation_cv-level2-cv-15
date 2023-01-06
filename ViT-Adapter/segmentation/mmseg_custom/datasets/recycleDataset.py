# from mmseg.datasets.builder import DATASETS
# from mmseg.datasets.custom import CustomDataset
# import os.path as osp


# classes = ("Background", "General trash", "Paper", "Paper pack", "Metal",
#         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# palette = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128],
#             [64, 0, 192], [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]]

# @DATASETS.register_module()
# class RecycleDataset(CustomDataset):
#     CLASSES = classes
#     PALETTE = palette
#     def __init__(self, split, **kwargs):
#         super().__init__(img_suffix='.png', seg_map_suffix='.png', 
#                         split=split, **kwargs)
#         assert osp.exists(self.img_dir) and self.split is not None

# # # @DATASETS.register_module()
# # # class RecycleDataset(CustomDataset):
# # #     """Custom dataset for semantic segmentation. An example of file structure
# # #     is as followed."""
# # #     classes = ("Background", "General trash", "Paper", "Paper pack", "Metal",
# # #         "Glass", "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
# # #     pallete = [[0, 0, 0], [192, 0, 128], [0, 128, 192], [0, 128, 64], [128, 0, 0], [64, 0, 128],
# # #                 [64, 0, 192], [192, 128, 64], [192, 192, 128], [64, 64, 128], [128, 0, 192]]
# # #     def __init__(self, split, **kwargs):
# # #         super().__init__(split, **kwargs)

# # from mmseg.datasets.builder import DATASETS
# # from mmseg.datasets.custom import CustomDataset


