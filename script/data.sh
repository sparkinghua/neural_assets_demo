data_name="rand"

python data_generator.py \
    --data-name ${data_name} \
    --samples 1024 \
    --dataset train \

python data_generator.py \
    --data-name ${data_name} \
    --samples 64 \
    --dataset valid \

python data_generator.py \
    --data-name ${data_name} \
    --samples 64 \
    --dataset test \