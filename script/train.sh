python main.py \
    --config.file-paths.data "rand" \
    --config.file-paths.scene "spot" \
    --config.file-paths.resume None \
    --config.file-paths.mode "train" \
    --config.log-display.log-interval 256 \
    --config.log-display.display-interval 128 \
    --config.log-display.checkpoint-interval 10 \
    --config.model-param.hidden-features 256 \
    --config.model-param.num-layers 4 \
    --config.model-param.tex-res 512 \
    --config.model-param.tex-channels 8 \
    --config.model-param.fourier-enc True\
    --config.model-param.L-pos 6 \
    --config.model-param.L-dir 6 \
    --config.optim.batch-size 2 \
    --config.optim.lr 1e-3 \
    --config.optim.weight-decay 1e-4 \
    --config.optim.max-iter 50 \
    --config.optim.warmup-iter 5 \
    --config.optim.early-stopping 5 \