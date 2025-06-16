import os

# 当前文件路径
current_file = os.path.abspath(__file__)
# 项目根目录：回到 src/ 上一层，即 project/
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

#目录路径
ROOT_DIR = project_root
RESULTS_DIR = os.path.join(project_root, "Results")
SRC_DIR = os.path.join(project_root, "src")
DATA_DIR = os.path.join(project_root, "data")
