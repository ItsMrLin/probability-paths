from pathlib import Path


class NBAConfig:
    NAME = "NBA"
    DATA_ROOT = Path("../../data/")
    INPUT_ROOT = DATA_ROOT / "input" / "nba"
    OUTPUT_ROOT = DATA_ROOT / "output" / "nba"

    TRAIN_PATH = INPUT_ROOT / "train_path.csv"
    TEST_PATH = INPUT_ROOT / "test_path.csv"
    
    SAMPLE_LR_PATH = OUTPUT_ROOT / "lr.rds" 
    SAMPLE_MQLSTM_PATH = OUTPUT_ROOT / "mqlstm.rds" 
   
    NUMERIC_FEATURES = [
        "intercept",
        "home_score",
        "score_margin",
        "home_win_rate",
        "visitor_win_rate"]
    NUMERIC_FEATURES_TEMPORAL = ["home_score", "score_margin", "pred_outcome"]
    NUMERIC_FEATURES_CONSTANT = ["home_win_rate", "visitor_win_rate"]
    CATEGORICAL_FEATURES = ['time']
    GROUNDTRUTH_RENAME = 'home_win'
    GROUNDTRUTH = ['target_outcome']
    PREDICTION = 'pred_outcome'
    ID = "GAME_ID"
    COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + \
        [ID] + GROUNDTRUTH + [PREDICTION]
    T = 48
    N_FOLD = 5
    CUTOFF = 24
    NO_FEATURES = 6


class WeatherConfig:
    NAME = "WEATHER"
    DATA_ROOT = Path("../../data/")
    INPUT_ROOT = DATA_ROOT / "input" / "weather"
    OUTPUT_ROOT = DATA_ROOT / "output" / "weather"

    TRAIN_PATH = INPUT_ROOT / "train_path.csv"
    TEST_PATH = INPUT_ROOT / "test_path.csv"
    
    SAMPLE_LR_PATH = OUTPUT_ROOT / "lr.rds" 
    SAMPLE_MQLSTM_PATH = OUTPUT_ROOT / "mqlstm.rds" 
 
    CATEGORICAL_FEATURES = [
        'Location',
        'RainToday',
        'rain_1day_ago',
        'rain_2day_ago',
        'rain_3day_ago',
        'month',
    ]
    NUMERIC_FEATURES = [
        'time',
        'MinTemp',
        'MaxTemp',
        'Rainfall',
        'Humidity9am',
        'Humidity3pm',
        'Pressure9am',
        'Pressure3pm',
        'Cloud9am',
        'Cloud3pm',
        'Temp9am',
        'Temp3pm',
    ]
    GROUNDTRUTH_RENAME = 'finally_rain'
    GROUNDTRUTH = ['target_outcome']
    PREDICTION = 'pred_outcome'
    ID = "obs"
    T = 7
    COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + \
        [ID] + GROUNDTRUTH + [PREDICTION]

    N_FOLD = 5
    CUTOFF = 0

    NO_FEATURES = 70

EPSILON = 1e-10
SANDBOX_DIR = str(Path("../../data/sandbox"))

