from __future__ import division
import numpy as np
from glob import glob
import os
import scipy.misc

class euroc_raw_loader(object):
    def __init__(self, 
                 dataset_dir,
                 img_height=7,
                 img_width=256,
                 seq_length=5):
        # 只读一个数据库吗？不是吧
        # 此文件所在的位置
        self.dataset_choice = ['MH_01_easy','MH_02_easy']
        # self.dataset_choice = ['MH_01_easy','MH_01_easy','MH_01_medium','MH_04_difficult','V1_01_easy']

        self.dataset_items = []
        self.camera_id = 0
        self.dataset_dir = dataset_dir
        self.img_height = img_height
        self.img_width = img_width
        self.seq_length = seq_length

        self.flex_num = 1
        # self.seq_length = 3

        self.timestamps = []
        self.filenames = []
        self.fullnames = []    
        
        for dataset in self.dataset_choice:
            one_dataset = os.path.join(dataset_dir, dataset)
            # dir_path = os.path.dirname(os.path.realpath(__file__))

            cam0_path = os.path.join(one_dataset,'mav0','cam' + str(self.camera_id))
            # cam1_path = os.path.join(dir_path,'mav0','cam1')
            
            cam0_csv = os.path.join(cam0_path,'data.csv')
            # cam1_csv = os.path.join(cam1_path,'data.csv')

            # 需要读取相机参数，这个我想就直接写在文件里面算了qwq
            cam0_sensor = os.path.join(cam0_path, 'sensor.yaml')

            # static_frames_file = dir_path + '/static_frames.txt'
            # test_scene_file = dir_path + '/test_scenes_' + split + '.txt'
            with open(cam0_csv, 'r') as f:
                timestamp_filename = f.readlines()

            # 读进去了之后就要处理这些文件，处理的方式是把这些东西三帧图像合成一个，为了简便起见我就不弄验证那个时间戳是不是对齐的问题了，直接连续三帧图像然后对齐。  

            # test_scenes = []
            # 读入的条目，里面是时间戳和文件名， 第一行因为是标题所以跳过
            for line in timestamp_filename[1:]:
                # test_scenes.append(line[:-1])
                line = line[:-1]
                item = line.split(',')
                timestamp = int(item[0])
                filename = str(item[1])
                self.timestamps.append(timestamp)
                self.filenames.append(filename)
                self.fullnames.append(os.path.join(cam0_path,'data',filename))
                # 唯一id
                self.dataset_items.append(dataset + '#cam'+ str(self.camera_id) +'#' + str(timestamp) +'#'+filename)

        # 得到了时间戳和文件名，需要验证组合是不是正确
        # 验证方法：如果连续id号当中，数据集不一样，则判断为错误，否则为正确。
        self.num_train = len(self.dataset_items)

    def is_valid_sample(self, n):
        # self.flex_num = 2
        target_item = self.dataset_items[n]
        current_dataset, _, _, _ = target_item.split('#')
        min_n = n - self.flex_num
        max_n = n + self.flex_num
        # 如果编号不在范围之内的话
        if min_n < 0 or max_n > len(self.dataset_items):
            return False
        # 检查在这一段范围内的编号
        for index in range(min_n, max_n):
            dataset, _ , _ , _ = self.dataset_items[index].split('#')
            if current_dataset != dataset:
                return False
        # print('正确！')
        return True

    # 这个函数必须实现，输出是所有图片的集合
    def get_train_example_with_idx(self, n):
        if not self.is_valid_sample(n):
            return False
        example = self.load_example(n)
        return example

    def load_example(self, n):
        image_seq, zoom_x, zoom_y = self.load_image_sequence(n)
        # tgt_drive, tgt_cid, tgt_frame_id = frames[n].split(' ')
        intrinsics = self.load_intrinsics_raw()
        intrinsics = self.scale_intrinsics(intrinsics, zoom_x, zoom_y)
        example = {}
        example['intrinsics'] = intrinsics
        example['image_seq'] = image_seq
        example['folder_name'] = os.path.dirname(self.fullnames[n]).split('/')[-4]
        example['file_name'] = self.filenames[n]
        # print(os.path.dirname(self.fullnames[n]))
        return example

    def load_image_sequence(self, n):
        min_n = n - self.flex_num
        # 因为python是左闭右开的
        max_n = n + self.flex_num + 1 
        image_seq = []
        for index in range(min_n, max_n):
            # curr_drive, curr_cid, curr_frame_id = self.dataset_items[index].split('#')
            # print(self.fullnames[index])
            curr_img = self.load_image_raw(self.fullnames[index])
            if index == n:
                zoom_y = self.img_height/curr_img.shape[0]
                zoom_x = self.img_width/curr_img.shape[1]
            curr_img = scipy.misc.imresize(curr_img, (self.img_height, self.img_width))
            image_seq.append(curr_img)
        return image_seq, zoom_x, zoom_y

    def load_image_raw(self, filename):
        # img_file = os.path.join(self.dataset_dir, date, drive, 'image_' + cid, 'data', frame_id + '.png')
        img = scipy.misc.imread(filename)

        return img

    def load_intrinsics_raw(self):
        # date = drive[:10]
        # calib_file = os.path.join(self.dataset_dir, date, 'calib_cam_to_cam.txt')

        # filedata = self.read_raw_calib_file(calib_file)
        # P_rect = np.reshape(filedata['P_rect_' + cid], (3, 4))
        # intrinsics = P_rect[:3, :3]
        intrinsics = np.array([[458.654, 0.0, 367.215],
            [0.0,457.296,248.375],
            [0.0,0.0,1.0]
        ],dtype=np.float)
        return intrinsics

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out
