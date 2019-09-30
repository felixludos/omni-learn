

CUDA_VISIBLE_DEVICES=1 python main_classify.py --config config/genre-yt.yaml --name crnn-wav-seq3000-rec2x256 --data-type wav --model crnn --seq-len 3000 --rec-dim 256 --rec-num-layers 2


CUDA_VISIBLE_DEVICES=3 python main_classify.py --config config/genre-yt.yaml --name test-full-song-crnn-wav-seq3000-rec2x256 --data-type wav --model crnn --seq-len 3000 --rec-dim 256 --rec-num-layers 2 --rec-type gru --dense-labeling --full-records --single-step


