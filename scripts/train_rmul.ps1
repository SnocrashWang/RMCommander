# 阶段A
# 只训练基本行为，包括靠近敌人、攻击目标等
# python train_rmul.py `
#     --blue-mode script `
#     --episodes 2000 `
#     --save-interval 200 `
#     --eval-interval 200 `
#     --eval-episodes 100
python train_rmul_parallel.py `
    --blue-mode script `
    --episodes 100 `
    --rollout-batch-size 100 `
    --num-workers 8 `
    --save-interval 10 `
    --eval-interval 10 `
    --eval-episodes 20 `
    --parallel-eval


# 阶段B
# 强化行为，将规则锁定在固定起始点
# python train_base_stage.py `
#     --stage b `
#     --episodes 2000 `
#     --save-interval 200 `
#     --eval-interval 200 `
#     --eval-episodes 100 `
#     --base-model "models\stage_a_agent_20260608_194956.pt"

# 阶段C
# 敌人行为多样化
# python train_base_stage.py `
#     --stage c `
#     --episodes 2000 `
#     --save-interval 200 `
#     --eval-interval 200 `
#     --eval-episodes 100 `
#     --base-model "models\stage_b_agent_20260608_203400_episode_1800.pt"
