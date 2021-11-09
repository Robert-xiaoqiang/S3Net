#! /usr/bin/bash
cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 350/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 300/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 250/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 200/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 150/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 100/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py

cd ~/projects/saliency/SCFNet/output/exp-reduce-channel-mt-0/SCFNet_Res50/pth/
cp 50/* ./
cd ~/projects/saliency/SCFNet/
python code/main_mt.py