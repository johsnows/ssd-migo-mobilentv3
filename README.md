# ssd-migo-mobilentv3-
train:
python train_ssd.py --datasets /userhome/VOCdevkit/VOC2007 /userhome/VOCdevkit/VOC2012 --validation_dataset /userhome/VOCdevkit/VOCtest2007  --net migo --scheduler cosine --lr 0.01 --t_max 200 --validation_epochs 5 --num_epochs 200 --structure_path structures/dynamic_sng_V3_600/dynamic_SNG_V3_ofa_imagenet_width_multi_1.3_600_B.json --migo_size 2
eval:
python eval_ssd.py --net migo  --dataset /userhome/VOCdevkit/VOCtest2007/ --trained_model models/(yourtrain.pth)  --label_file models/voc-model-labels.txt --structure_path structures/dynamic_sng_V3_600/dynamic_SNG_V3_ofa_imagenet_width_multi_1.3_600_B.json --migo_size 2 --eval_dir eval_results/migo/600/gpu/0
