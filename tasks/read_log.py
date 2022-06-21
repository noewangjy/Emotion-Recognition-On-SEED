import os
import glob
import numpy as np


log_path = "DAN/multirun/2022-06-06/21-40-27"

acc_list = []

for i in range(15):
    log_dir = os.path.join(log_path, str(i) + "/checkpoints")
    ckpt_files = glob.glob(os.path.join(log_dir, "*.ckpt"))
    max_acc: float = 0.0
    for ckpt in ckpt_files:
        if ckpt.split("/")[-1].startswith("epoch"):

            current_acc = float(ckpt.split("--")[-1].split("=")[-1][:-5])

            max_acc = max(current_acc, max_acc)
    acc_list.append(max_acc)

scores = np.array(acc_list)

print(scores)
print(f"Acc.mean = {np.mean(scores)}, Acc.std = {np.std(scores)}")




