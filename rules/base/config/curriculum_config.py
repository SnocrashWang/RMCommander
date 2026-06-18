from rules.base.curriculum import *

# 不同阶段的课程设计
# 键值元组的元素分别代表 (ratio, (random_env, random_obstacles, random_robot))
BASE_CURRICULUM_LIST = [
    {
        CurriculumBaseMovement: (0.7, (True, True, True)),
        CurriculumBaseBattle: (0.2, (True, True, True)),
        CurriculumBaseEasy: (0.1, (True, True, True)),
    },
    {
        CurriculumBaseMovement: (0.2, (True, True, True)),
        CurriculumBaseBattle: (0.7, (True, True, True)),
        CurriculumBaseEasy: (0.1, (True, True, True)),
    },
    {
        CurriculumBaseMovement: (0.2, (True, True, True)),
        CurriculumBaseBattle: (0.2, (True, True, True)),
        CurriculumBaseEasy: (0.6, (True, True, True)),
    },
    {
        CurriculumBaseMovement: (0.1, (True, True, True)),
        CurriculumBaseBattle: (0.1, (True, True, True)),
        CurriculumBaseEasy: (0.2, (True, True, True)),
        CurriculumBaseMedium: (0.6, (True, True, True)),
    },
    {
        CurriculumBaseEasy: (0.5, (True, True, True)),
        CurriculumBaseMedium: (0.3, (True, True, True)),
        CurriculumBaseHard: (0.2, (True, True, True)),
    },
    {
        CurriculumBaseEasy: (0.1, (True, True, True)),
        CurriculumBaseMedium: (0.3, (True, True, True)),
        CurriculumBaseHard: (0.6, (True, True, True)),
    },
    {
        CurriculumBaseMedium: (0.1, (True, True, True)),
        CurriculumBaseHard: (0.9, (True, True, True)),
    },
]

BASE_CURRICULUM_EVAL = {
    CurriculumBaseHard: (1.0, (False, True, False))
}
