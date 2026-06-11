PARAMS=(
    # --base-model ""
    --model-dir "models/rmul"
    --log-dir "logs"
    --control-frequency 2
    --blue-mode "script"
    --episodes 100
    --rollout-batch-size 4
    --num-workers 2
    --save-interval 10
    --eval-interval 10
    --eval-episodes 20
    --parallel-eval
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
