from enum import Enum

LEVEL_NEED_EXP = {
    1: 0,
    2: 400,
    3: 800,
    4: 1200,
    5: 1600,
    6: 2000,
    7: 2400,
    8: 2800,
    9: 3200,
    10: 4000,
}

class CHASSIS_PROPERTY_TYPE(Enum):
    POWER = "POWER"
    HP = "HP"
    DEFAULT = "DEFAULT"

CHASSIS_PROPERTY_HERO = {
    CHASSIS_PROPERTY_TYPE.POWER: {
        1: {"HP": 200, "POWER": 70},
        2: {"HP": 225, "POWER": 75},
        3: {"HP": 250, "POWER": 80},
        4: {"HP": 275, "POWER": 85},
        5: {"HP": 300, "POWER": 90},
        6: {"HP": 325, "POWER": 95},
        7: {"HP": 350, "POWER": 100},
        8: {"HP": 375, "POWER": 105},
        9: {"HP": 400, "POWER": 110},
        10: {"HP": 500, "POWER": 120},
    },
    CHASSIS_PROPERTY_TYPE.HP: {
        1: {"HP": 250, "POWER": 55},
        2: {"HP": 275, "POWER": 60},
        3: {"HP": 300, "POWER": 65},
        4: {"HP": 325, "POWER": 70},
        5: {"HP": 350, "POWER": 75},
        6: {"HP": 375, "POWER": 80},
        7: {"HP": 400, "POWER": 85},
        8: {"HP": 425, "POWER": 90},
        9: {"HP": 450, "POWER": 100},
        10: {"HP": 500, "POWER": 120},
    },
    CHASSIS_PROPERTY_TYPE.DEFAULT: {
        i: {"HP": 150, "POWER": 50} for i in range(1, 11)
    }
}

CHASSIS_PROPERTY_STANDARD = {
    CHASSIS_PROPERTY_TYPE.POWER: {
        1: {"HP": 150, "POWER": 60},
        2: {"HP": 175, "POWER": 65},
        3: {"HP": 200, "POWER": 70},
        4: {"HP": 225, "POWER": 75},
        5: {"HP": 250, "POWER": 80},
        6: {"HP": 275, "POWER": 85},
        7: {"HP": 300, "POWER": 90},
        8: {"HP": 325, "POWER": 95},
        9: {"HP": 350, "POWER": 100},
        10: {"HP": 400, "POWER": 100},
    },
    CHASSIS_PROPERTY_TYPE.HP: {
        1: {"HP": 200, "POWER": 45},
        2: {"HP": 225, "POWER": 50},
        3: {"HP": 250, "POWER": 55},
        4: {"HP": 275, "POWER": 60},
        5: {"HP": 300, "POWER": 65},
        6: {"HP": 325, "POWER": 70},
        7: {"HP": 350, "POWER": 75},
        8: {"HP": 375, "POWER": 80},
        9: {"HP": 400, "POWER": 90},
        10: {"HP": 400, "POWER": 100},
    },
    CHASSIS_PROPERTY_TYPE.DEFAULT: {
        i: {"HP": 100, "POWER": 40} for i in range(1, 11)
    }
}

class GIMBAL_PROPERTY_TYPE(Enum):
    HEAT = "HEAT"
    COOL_DOWN = "COOL_DOWN"
    DEFAULT = "DEFAULT"

GIMBAL_PROPERTY_17 = {
    GIMBAL_PROPERTY_TYPE.HEAT: {
        1: {"HEAT": 200, "COOL_DOWN": 10},
        2: {"HEAT": 250, "COOL_DOWN": 15},
        3: {"HEAT": 300, "COOL_DOWN": 20},
        4: {"HEAT": 350, "COOL_DOWN": 25},
        5: {"HEAT": 400, "COOL_DOWN": 30},
        6: {"HEAT": 450, "COOL_DOWN": 35},
        7: {"HEAT": 500, "COOL_DOWN": 40},
        8: {"HEAT": 550, "COOL_DOWN": 45},
        9: {"HEAT": 600, "COOL_DOWN": 50},
        10: {"HEAT": 650, "COOL_DOWN": 60},
    },
    GIMBAL_PROPERTY_TYPE.COOL_DOWN: {
        1: {"HEAT": 50, "COOL_DOWN": 40},
        2: {"HEAT": 85, "COOL_DOWN": 45},
        3: {"HEAT": 120, "COOL_DOWN": 50},
        4: {"HEAT": 155, "COOL_DOWN": 55},
        5: {"HEAT": 190, "COOL_DOWN": 60},
        6: {"HEAT": 225, "COOL_DOWN": 65},
        7: {"HEAT": 260, "COOL_DOWN": 70},
        8: {"HEAT": 295, "COOL_DOWN": 75},
        9: {"HEAT": 330, "COOL_DOWN": 80},
        10: {"HEAT": 400, "COOL_DOWN": 80},
    },
    GIMBAL_PROPERTY_TYPE.DEFAULT: {
        i: {"HEAT": 40, "COOL_DOWN": 100} for i in range(1, 11)
    }
}

GIMBAL_PROPERTY_42 = {
    GIMBAL_PROPERTY_TYPE.DEFAULT: {
        1: {"HEAT": 100, "COOL_DOWN": 40},
        2: {"HEAT": 140, "COOL_DOWN": 48},
        3: {"HEAT": 180, "COOL_DOWN": 56},
        4: {"HEAT": 220, "COOL_DOWN": 64},
        5: {"HEAT": 260, "COOL_DOWN": 72},
        6: {"HEAT": 300, "COOL_DOWN": 80},
        7: {"HEAT": 340, "COOL_DOWN": 88},
        8: {"HEAT": 380, "COOL_DOWN": 96},
        9: {"HEAT": 420, "COOL_DOWN": 104},
        10: {"HEAT": 500, "COOL_DOWN": 120},
    },
}
