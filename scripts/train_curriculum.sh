PARAMS=(
    --base-model "models\base\ppo_agent_20260616_235752_stage_4.pt"
    --model-dir "models/base"
    --control-frequency 2
    --curriculum-stage 5
    --episodes 200
    --rollout-batch-size 50
    --num-workers 4
    --save-interval 20
    --eval-interval 20
    --eval-episodes 20
    # --deterministic-eval
    --device "cuda:0"
    --actor-lr 5e-5
    --critic-lr 5e-4
    --alpha 0.01
    --gamma 0.98
    --lmbda 0.95
    --epochs 4
    --eps 0.1
    --ppo-batch-size 16
)

python train_curriculum_parallel.py "${PARAMS[@]}"
