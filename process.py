import os
import numpy
from PIL import Image
dp = "./rgb_data/"
folders = sorted(os.listdir(dp))
print(folders)

for i in range(4):
    folder = folders[i]
    fns = sorted(os.listdir(dp+folder))
    for j in range(6):
        if j <5:
            with open("./rgb_train.txt","a") as f:
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4], dp+folder+'/'+fns[j*4+1], 0))
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+1], dp+folder+'/'+fns[j*4+2], 0))
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+2], dp+folder+'/'+fns[j*4+3], 1))
        else:
            with open("./rgb_test.txt","a") as f:
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4], dp+folder+'/'+fns[j*4+1], 0))
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+1], dp+folder+'/'+fns[j*4+2], 0))
                f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+2], dp+folder+'/'+fns[j*4+3], 1))

i = 4
folder = folders[i]
fns = sorted(os.listdir(dp+folder))
for j in range(6):
        with open("./rgb_test.txt","a") as f:
            f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4], dp+folder+'/'+fns[j*4+1], 0))
            f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+1], dp+folder+'/'+fns[j*4+2], 0))
            f.write("{} {} {}\n".format(dp+folder+'/'+fns[j*4+2], dp+folder+'/'+fns[j*4+3], 1))

