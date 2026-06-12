# 阶段A
# 只训练基本行为，包括靠近敌人、攻击目标等
python train_base_stage.py \
    --stage a \
    --episodes 1000 \
    --save-interval 200 \
    --eval-interval 200 \
    --eval-episodes 20 \
    --random-start

# 阶段B
# 强化行为，将规则锁定在固定起始点
# python train_base_stage.py \
#     --stage b \
#     --episodes 2000 \
#     --save-interval 200 \
#     --eval-interval 200 \
#     --eval-episodes 20 \
#     --base-model "models\base\ppo_agent_20260612_184137_stage_a.pt"

# 阶段C
# 敌人行为多样化
# python train_base_stage.py \
#     --stage c \
#     --episodes 2000 \
#     --save-interval 200 \
#     --eval-interval 200 \
#     --eval-episodes 100 \
#     --base-model "models\stage_b_agent_20260608_203400_episode_1800.pt"
