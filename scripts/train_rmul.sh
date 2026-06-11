PARAMS=(
    # --base-model ""
    --model-dir "models/rmul"
    --log-dir "logs"
    --control-frequency 2
    --blue-mode "script"
    --episodes 100
    --rollout-batch-size 50
    --num-workers 16
    --save-interval 5
    --eval-interval 5
    --eval-episodes 20
    --parallel-eval
    # --deterministic-eval
    # --device "cuda:0"
    --actor-lr 5e-5
    --critic-lr 5e-4
    --gamma 0.98
    --lmbda 0.95
    --epochs 4
    --eps 0.1
    --ppo-batch-size 16
)

python train_rmul_parallel.py "${PARAMS[@]}"
