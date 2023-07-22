import os
import sys
from datetime import datetime
from gym.envs.registration import register

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "yes please"
LOG_DIR = os.path.join(os.getcwd(), "logs")
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

# Init and setup the root logger
#logging.basicConfig(filename=LOG_DIR + '/macad-gym.log', filemode='w', level=logging.DEBUG)
# Set this where you want to save image outputs (or empty string to disable)
LOG_PATH = os.path.join(LOG_DIR, f"{datetime.today().strftime('%Y-%m-%d_%H-%M')}")
#CARLA_OUT_PATH = os.environ.get("CARLA_OUT", os.path.expanduser("~/Git/RLAV_in_Carla_Gym/carla_out"))
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)

# Fix path issues with included CARLA API
sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "carla/PythonAPI"))

# Set this to the path of your Carla binary
SERVER_BINARY = os.environ.get(
    "CARLA_SERVER", os.path.expanduser("~/ProgramFiles/Carla/CarlaUE4.sh")
)
assert os.path.exists(SERVER_BINARY), (
    "Make sure CARLA_SERVER environment"
    " variable is set & is pointing to the"
    " CARLA server startup script (Carla"
    "UE4.sh). Refer to the README file/docs."
)

# Check if is using on Windows
IS_WINDOWS_PLATFORM = "win" in sys.platform

# Declare available environments with a brief description
_AVAILABLE_ENVS = {
    'HomoNcomIndePOIntrxMASS3CTWN3-v0': {
        "entry_point": "macad_gym.envs:HomoNcomIndePOIntrxMASS3CTWN3",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Intersection Multi-Agent scenario with "
        "Stop-Sign, 3 Cars in Town3, version 0"
    },
    "HomoNcomIndePoHiwaySAFR2CTWN5-v0": {
        "entry_point": "macad_gym.envs:HomoNcomIndePoHiwaySAFR2CTWN5",
        "description":
        "Homogeneous, Non-communicating, Independent, Partially-"
        "Observable Hiway Multi-Agent scenario with "
        "Traffic Lights and Fixed Route, 2 Cars in Town5, version 0"
    },
    'HeteNcomIndePOIntrxMATLS1B2C1PTWN3-v0': {
        "entry_point": "macad_gym.envs:HeteNcomIndePOIntrxMATLS1B2C1PTWN3",
        "description":
        "Heterogeneous, Non-communicating, Independent,"
        "Partially-Observable Intersection Multi-Agent"
        " scenario with Traffic-Light Signal, 1-Bike, 2-Car,"
        "1-Pedestrian in Town3, version 0"
    }
}

for env_id, val in _AVAILABLE_ENVS.items():
    register(id=env_id, entry_point=val.get("entry_point"))


def list_available_envs():
    print("Environment-ID: Short-description")
    import pprint
    available_envs = {}
    for env_id, val in _AVAILABLE_ENVS.items():
        available_envs[env_id] = val.get("description")
    pprint.pprint(available_envs)
