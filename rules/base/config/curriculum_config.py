from rules.base.curriculum import *

# 不同阶段的课程设计
# 键值元组的元素分别代表 (ratio, (random_env, random_obstacles, random_robot))
CURRICULUM_LIST_BASE = [
    {
        CurriculumBaseMovement: (0.7, (True, True, True)),
        CurriculumBaseBattle: (0.2, (True, True, True)),
        CurriculumBaseEasy: (0.1, (True, True, False)),
    },
    {
        CurriculumBaseMovement: (0.2, (True, True, True)),
        CurriculumBaseBattle: (0.7, (True, True, True)),
        CurriculumBaseEasy: (0.1, (True, True, False)),
    },
    {
        CurriculumBaseMovement: (0.2, (True, True, True)),
        CurriculumBaseBattle: (0.2, (True, True, True)),
        CurriculumBaseEasy: (0.6, (True, True, False)),
    },
    {
        CurriculumBaseMovement: (0.1, (True, True, True)),
        CurriculumBaseBattle: (0.1, (True, True, True)),
        CurriculumBaseEasy: (0.2, (True, True, False)),
        CurriculumBaseMedium: (0.6, (True, True, True)),
    },
]

CURRICULUM_EVAL_BASE = {
    CurriculumBaseHard: (1.0, (False, True, False))
}
