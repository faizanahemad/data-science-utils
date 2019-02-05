import sys, os

sys.path.append(os.getcwd())

import os.path
import sys
from inspect import getsourcefile

current_path = os.path.abspath(getsourcefile(lambda:0))
current_dir = os.path.dirname(current_path)
parent_dir = current_dir[:current_dir.rfind(os.path.sep)]

sys.path.insert(0, parent_dir)

from misc import save_list_per_line
from misc import load_list_per_line

# arr = [["aa","bb","cc"],["dd","ee","ff"]]
arr = ["terry has a teddy bear","brown bears are just like teddy bears"]
save_list_per_line(arr,"output.txt")

print(load_list_per_line("output.txt"))