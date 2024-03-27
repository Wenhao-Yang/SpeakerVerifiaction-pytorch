# Copyright (c) 2020 Mobvoi Inc (Di Wu)
#               2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import glob
import re
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(description='average model')
    parser.add_argument('--check-path', required=True,
                        help='src model path for average')
    parser.add_argument('--num', default=3, type=int,
                        help='nums for averaged model')
    parser.add_argument('--min_epoch', default=0, type=int,
                        help='min epoch used for averaging model')
    parser.add_argument('--max_epoch',
                        default=65536,  # Big enough
                        type=int,
                        help='max epoch used for averaging model')
    
    args = parser.parse_args()
    print(args)
    return args


def main():
    args = get_args()
    num = args.num

    # path_list = glob.glob('{}/[!avg][!final][!convert]*.pt'.format(
    #     args.src_path))
    path_list = glob.glob('{}/checkpoint*[!avg][!final][!convert].pth'.format(args.check_path))

    # path_list = sorted(
    #     path_list,
    #     key=lambda p: int(re.findall(r"(?<=model_)\d*(?=.pt)", p)[0]))
    
    path_list = sorted(path_list, key=lambda p: int(re.split('\_|\.', p.split('/')[-1])[1]))
    epochs = sorted([int(re.split('\_|\.', p.split('/')[-1])[1]) for p in path_list])
    path_list = path_list[-num:]
    epochs = epochs[-num:]
    print('average epoch ckp: ', epochs)

    avg = None
    assert num == len(path_list), print(num, path_list)
    for path in path_list:
        # print('Processing {}'.format(path))
        states = torch.load(path, map_location=torch.device('cpu'))
        states = states['state_dict'] if 'state_dict' in states else states
        if avg is None:
            avg = states
        else:
            for k in avg.keys():
                avg[k] += states[k]
    # average
    for k in avg.keys():
        if avg[k] is not None:
            # pytorch 1.6 use true_divide instead of /=
            avg[k] = torch.true_divide(avg[k], num)

    save_model = os.path.join(args.check_path, 'checkpoint_avg{}.pth'.format(num))
    print('Saving to {}'.format(save_model))
    
    torch.save({'epoch': epochs, 'state_dict': avg} , save_model)


if __name__ == '__main__':
    main()
