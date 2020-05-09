#! /usr/bin/bash
cd /home/xqwang/projects/saliency/SCFNet/output/exp-reduce-channel-ss-2/S3CFNet_Res50/pth
mkdir final/
cp *.pth* final/
cp best/* ./
cd -
python code/main_ss.py
cd /home/xqwang/projects/saliency/SCFNet/output/exp-reduce-channel-ssmt-2/S3CFNet_Res50/pth
mkdir final/
cp *.pth* final/
cp best/* ./
cd -
python code/main_ss_mt.py
