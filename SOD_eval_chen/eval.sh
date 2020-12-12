datasets=(NJU2K NLPR STERE DES LFSD SSD SIP)

dir=~/codes/BBS-Net/BBS-Net/test_maps/BBSNet_epoch_150
method=BBSNet_epoch_150

for i in {0..6}
do
	dataset=${datasets[${i}]}
	echo "Moving ${dataset}..."
	mkdir -p results/pred/${method}/${dataset}/all
	mv ${dir}/${dataset}/*png results/pred/${method}/${dataset}/all/
	# read -p var
done

echo -e '\n' >> results/result.txt
CUDA_VISIBLE_DEVICES=0 python main.py --methods ${method} --save_dir ./results --root_dir ./results --datasets ${datasets[0]}+${datasets[1]}+${datasets[2]}+${datasets[3]}+${datasets[4]}+${datasets[5]}+${datasets[6]}

echo "Finished..."


