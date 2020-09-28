#!/bin/bash
rm -rf out
rm result.txt
touch result.txt

echo "crf $1 resolution $2 x $3"

for input in ../videos_crop/*.yuv; do

	echo "orginal"

	echo "-------------------"

	echo "$input"

	mkdir -p out/source/

	ffmpeg -pix_fmt yuv420p -s:v $2x$3 -i $input -f image2 out/source/img%06d.png

	mkdir -p out/h265/

	FFREPORT=file=ffreport.log:level=56 ffmpeg -pix_fmt yuv420p -s $2x$3 -i $input -c:v libx265 -tune zerolatency -x265-params "crf=$1:keyint=12:verbose=1" out/h265/out.mkv

	ffmpeg -i out/h265/out.mkv -f image2 out/h265/img%06d.png

    mkdir -p ${input%.*}/H265L$1/
	ffmpeg -i out/h265/out.mkv -f image2 ${input%.*}/H265L$1/im%04d.png
    echo $input

	CUDA_VISIBLE_DEVICES=1 python3 measure265.py $input $2 $3  >> result.txt

	rm -rf out/h265

	rm ffreport.log

	rm -rf out/source

	echo "-------------------"

done

python3 report.py
