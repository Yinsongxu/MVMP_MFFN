from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import glob
import re

from data.datasets import ImageDataset


class MSMT17(ImageDataset):

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = ''
        #self.download_dataset(self.dataset_dir, self.dataset_url)
        self.train_dir = '/home/mtc-206/msmt/bounding_box_train'
        self.query_dir = '/home/mtc-206/msmt/query'
        self.gallery_dir = '/home/mtc-206/msmt/bounding_box_test'
        
        required_files = [
            
            self.train_dir,
            self.query_dir,
            self.gallery_dir
        ]
        self.check_before_run(required_files)

        train = self.process_dir(self.train_dir, relabel=True)
        query = self.process_dir(self.query_dir, relabel=False)
        gallery = self.process_dir(self.gallery_dir, relabel=False)

        super(MSMT17, self).__init__(train, query, gallery, **kwargs)

    def process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid:label for label, pid in enumerate(pid_container)}

        data = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            
            camid -= 1 # index starts from 0
            if relabel: pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data