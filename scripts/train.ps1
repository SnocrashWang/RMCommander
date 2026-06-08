# 阶段A
# 只训练基本行为，包括靠近敌人、攻击目标等
# python stage_diagnostic.py `
#     --stage a `
#     --episodes 1000 `
#     --save-interval 200 `
#     --eval-interval 200 `
#     --eval-episodes 50 `
#     --random-start

# 阶段B
# 强化行为，将规则锁定在固定起始点
# python stage_diagnostic.py `
#     --stage b `
#     --episodes 2000 `
#     --save-interval 200 `
#     --eval-interval 200 `
#     --eval-episodes 50 `
#     --base-model "models\stage_a_agent_20260608_194956.pt"

# 阶段C
# 敌人行为多样化
python stage_diagnostic.py `
    --stage c `
    --episodes 2000 `
    --save-interval 200 `
    --eval-interval 200 `
    --eval-episodes 50 `
    --base-model "models\stage_b_agent_20260608_203400_episode_1800.pt"
