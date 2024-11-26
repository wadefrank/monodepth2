# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import os
import argparse

file_dir = os.path.dirname(__file__)  # the directory that options.py resides in


class MonodepthOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Monodepthv2 options")

        ### PATHS 路径设置
        # 训练数据集路径，默认为：./kitti_data
        self.parser.add_argument("--data_path",
                                 type=str,
                                 help="path to the training data",
                                 default=os.path.join(file_dir, "kitti_data"))
        # 日志输出路径，默认为：/home/user_name/tmp
        # os.path.expanduser("~")：返回当前用户的主目录路径
        self.parser.add_argument("--log_dir",
                                 type=str,
                                 help="log directory",
                                 default=os.path.join(os.path.expanduser("~"), "tmp")) 

        ### TRAINING options 训练选项
        # 保存模型日志的文件夹的名称，如：mono_model， stereo_model 以及mono+stereo_model，默认为：mdp
        self.parser.add_argument("--model_name",
                                 type=str,
                                 help="the name of the folder to save the model in",
                                 default="mdp")
        # 使用哪个训练分组，默认为：eigen_zhou
        self.parser.add_argument("--split",
                                 type=str,
                                 help="which training split to use",
                                 choices=["eigen_zhou", "eigen_full", "odom", "benchmark", "fx"],
                                 default="eigen_zhou")
        # resnet层的数量，默认为18
        self.parser.add_argument("--num_layers",
                                 type=int,
                                 help="number of resnet layers",
                                 default=18,
                                 choices=[18, 34, 50, 101, 152])
        # 要训练哪种类型的数据集，默认为kitti
        self.parser.add_argument("--dataset",
                                 type=str,
                                 help="dataset to train on",
                                 default="kitti",
                                 choices=["kitti", "kitti_odom", "kitti_depth", "kitti_test"])
        # 如果设置，则从原始 KITTI png 文件（而不是 jpg文件）训练，默认不设置
        self.parser.add_argument("--png",
                                 help="if set, trains from raw KITTI png files (instead of jpgs)",
                                 action="store_true")
        # 输入图像的高度，默认为：192
        self.parser.add_argument("--height",
                                 type=int,
                                 help="input image height",
                                 default=192)
        # 输入图像的宽度，默认为：640
        self.parser.add_argument("--width",
                                 type=int,
                                 help="input image width",
                                 default=640)
        # 视差平滑权重，默认为：0.001
        self.parser.add_argument("--disparity_smoothness",
                                 type=float,
                                 help="disparity smoothness weight",
                                 default=1e-3)
        # 其含义是在encoder和decoder时进行4级缩小和放大的多尺度，其倍数[0,1,2,3]分别对应为1, 2, 4, 8
        self.parser.add_argument("--scales",
                                 nargs="+",  # 表示这个参数后面可以跟一个或多个值
                                 type=int,
                                 help="scales used in the loss",
                                 default=[0, 1, 2, 3])
        # 最小深度，默认为：0.1
        self.parser.add_argument("--min_depth",
                                 type=float,
                                 help="minimum depth",
                                 default=0.1)
        # 最大深度，默认为：100
        self.parser.add_argument("--max_depth",
                                 type=float,
                                 help="maximum depth",
                                 default=100.0)
        # 如果设置，则使用stereo pair进行训练，默认不设置
        self.parser.add_argument("--use_stereo",
                                 help="if set, uses stereo pair for training",
                                 action="store_true")
        # 要加载的帧 0代表当前输入的样本图片，-1则代表当前帧在这个视频系列中的上一帧，1则代表下一帧
        self.parser.add_argument("--frame_ids",
                                 nargs="+",
                                 type=int,
                                 help="frames to load",
                                 default=[0, -1, 1])

        ### OPTIMIZATION options 优化选项
        # 批量大小，默认为：12
        self.parser.add_argument("--batch_size",
                                 type=int,
                                 help="batch size",
                                 default=12)
        # 学习率，默认为：0.0001
        self.parser.add_argument("--learning_rate",
                                 type=float,
                                 help="learning rate",
                                 default=1e-4)
        # 迭代次数，默认为：20
        self.parser.add_argument("--num_epochs",
                                 type=int,
                                 help="number of epochs",
                                 default=20)
        # 学习率调整的步长，默认为：15
        self.parser.add_argument("--scheduler_step_size",
                                 type=int,
                                 help="step size of the scheduler",
                                 default=15)

        ### ABLATION options 消融选项
        # 如果设置，使用monodepthv1多尺度，默认不设置
        self.parser.add_argument("--v1_multiscale",
                                 help="if set, uses monodepth v1 multiscale",
                                 action="store_true")
        # 如果设置，则使用平均重投影损失，默认不设置
        self.parser.add_argument("--avg_reprojection",
                                 help="if set, uses average reprojection loss",
                                 action="store_true")
        # 如果设置，则不进行automasking，默认不设置
        self.parser.add_argument("--disable_automasking",
                                 help="if set, doesn't do auto-masking",
                                 action="store_true")
        # 如果设置，则使用 Zhou 等人的预测掩码方案，默认不设置
        self.parser.add_argument("--predictive_mask",
                                 help="if set, uses a predictive masking scheme as in Zhou et al",
                                 action="store_true")
        # 如果设置，在损失中禁用 ssim，默认不设置
        self.parser.add_argument("--no_ssim",
                                 help="if set, disables ssim in the loss",
                                 action="store_true")
        # 选择 预训练 / 从头开始，默认使用预训练
        self.parser.add_argument("--weights_init",
                                 type=str,
                                 help="pretrained or scratch",
                                 default="pretrained",
                                 choices=["pretrained", "scratch"])
        # 位姿网络得到多少张图像，默认使用pairs
        self.parser.add_argument("--pose_model_input",
                                 type=str,
                                 help="how many images the pose network gets",
                                 default="pairs",
                                 choices=["pairs", "all"])
        # 位姿网络的类型：正常或共享
        self.parser.add_argument("--pose_model_type",
                                 type=str,
                                 help="normal or shared",
                                 default="separate_resnet",
                                 choices=["posecnn", "separate_resnet", "shared"])

        ### SYSTEM options 系统选项
        # 如果设置，则不使用CUDA，默认不设置
        self.parser.add_argument("--no_cuda",
                                 help="if set disables CUDA",
                                 action="store_true")
        # dataloader的线程数量，默认为：12
        self.parser.add_argument("--num_workers",
                                 type=int,
                                 help="number of dataloader workers",
                                 default=12)

        ### LOADING options 加载选项
        # 要加载的权重文件夹路径，可以用 ~ 表示当前用户的home路径
        self.parser.add_argument("--load_weights_folder",
                                 type=str,
                                 help="name of model to load")
        # 加载的模型
        self.parser.add_argument("--models_to_load",
                                 nargs="+",
                                 type=str,
                                 help="models to load",
                                 default=["encoder", "depth", "pose_encoder", "pose"])

        ### LOGGING options 日志选项
        # 每个tensorboard日志之间的batch，默认为：250
        self.parser.add_argument("--log_frequency",
                                 type=int,
                                 help="number of batches between each tensorboard log",
                                 default=250)
        # 每次保存之间的 epoch 数，默认为：1
        self.parser.add_argument("--save_frequency",
                                 type=int,
                                 help="number of epochs between each save",
                                 default=1)

        ### EVALUATION options 评估选项
        # 如果设置，则在stereo模式下评估，默认不设置
        self.parser.add_argument("--eval_stereo",
                                 help="if set evaluates in stereo mode",
                                 action="store_true")
        # 如果设置，则在mono模式下评估，默认不设置
        self.parser.add_argument("--eval_mono",
                                 help="if set evaluates in mono mode",
                                 action="store_true")
        # 如果设置，则在评估中禁用中值缩放，默认不设置
        self.parser.add_argument("--disable_median_scaling",
                                 help="if set disables median scaling in evaluation",
                                 action="store_true")
        # 如果设置，将预测乘以这个数字，默认为：1
        self.parser.add_argument("--pred_depth_scale_factor",
                                 help="if set multiplies predictions by this number",
                                 type=float,
                                 default=1)
        # 要评估的 .npy 差异文件的可选路径
        self.parser.add_argument("--ext_disp_to_eval",
                                 type=str,
                                 help="optional path to a .npy disparities file to evaluate")
        # 在哪个拆分上运行评估
        self.parser.add_argument("--eval_split",
                                 type=str,
                                 default="eigen",
                                 choices=[
                                    "eigen", "eigen_benchmark", "benchmark", "odom_9", "odom_10"],
                                 help="which split to run eval on")
        # 如果设置，保存预测的差异，默认不设置
        self.parser.add_argument("--save_pred_disps",
                                 help="if set saves predicted disparities",
                                 action="store_true")
        # 如果设置，则禁止评估，默认不设置
        self.parser.add_argument("--no_eval",
                                 help="if set disables evaluation",
                                 action="store_true")
        # 假设我们正在从 npy 加载特征结果，但我们想使用新的基准进行评估时设置
        self.parser.add_argument("--eval_eigen_to_benchmark",
                                 help="if set assume we are loading eigen results from npy but "
                                      "we want to evaluate using the new benchmark.",
                                 action="store_true")
        # 如果设置，会将差异输出到此文件夹
        self.parser.add_argument("--eval_out_dir",
                                 help="if set will output the disparities to this folder",
                                 type=str)
        # 如果设置，将从原始monodepth执行翻转后处理，默认不设置
        self.parser.add_argument("--post_process",
                                 help="if set will perform the flipping post processing "
                                      "from the original monodepth paper",
                                 action="store_true")

    def parse(self):
        self.options = self.parser.parse_args()
        return self.options
