PARAMS=(
    # --base-model ""
    --model-dir "models/base"
    --control-frequency 2
    --curriculum-stage 0
    --episodes 100
    --rollout-batch-size 50
    --num-workers 4
    --save-interval 10
    --eval-interval 10
    --eval-episodes 20
    # --deterministic-eval
    --device "cuda:0"
    --actor-lr 1e-4
    --critic-lr 5e-4
    --alpha 0.02
    --gamma 0.98
    --lmbda 0.95
    --epochs 4
    --eps 0.1
    --ppo-batch-size 16
)

python train_curriculum_parallel.py "${PARAMS[@]}"
