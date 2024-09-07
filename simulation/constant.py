import pandas as pd
import numpy as np
from datetime import datetime

START_DATE = datetime.strptime(f'2022-07-01 00:00', '%Y-%m-%d %H:%M')

# Efficiency, originally denoted by the Greek letter η,
# measures the efficiency of energy conversions and energy transfers.
ETA = 0.9

# The car's charging slows down from a battery state of charge (soc) of 85%.
SOC_OPT_THRESHOLD = 0.85  # for AC loading

CHARGING_INTERVAL = 1

MIN_CHARGING_CURRENT = 2.5

MIN_CHARGING_TIME = 15

MAX_CHARGING_TIME = 24 * 60 - 1

MAX_PV_POWER = 180000  # in W

MIN_ENERGY = 0  # in W
MIN_BATTERY_CAP = 8800  # in W
MAX_BATTERY_CAP = 89000  # in W

MIN_CHARGING_POWER = 3700
MAX_CHARGING_POWER = 22000

TRAIN_SPLIT = 0.8
VAL_SPLIT = 0.1
TEST_SPLIT = 0.1

GMM_SEED = 42
