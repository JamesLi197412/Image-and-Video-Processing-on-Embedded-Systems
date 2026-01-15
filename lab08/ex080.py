# git clone https://github.com/mrnugget/opencv-haar-classifier-training
# find ./positive_images -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) > positives.txt
# find ./negative_images -iname "*.jpeg" > negatives.txt

# perl bin/createsamples.pl positives.txt negatives.txt samples 1500 "opencv_createsamples -bgcolor 0 -bgthresh 0 --maxxangle 7 -maxyangle 7 -maxzangle 0.5 -maxidev 40 -w 120 -h 20"
# python ./tools/mergevec.py -v samples/ -o samples.vec
# opencv_traincascade -data classifier -vec samples.vec -bg negatives.txt -numStages 20 -minHitRate 0 .999 -maxFalseAlarmRate 0 .5 -numPos 1000 -numNeg 600 -w 120 -h 20 -mode ALL -precalcValBufSize 1024 -precalcIdxBufSize 1024
