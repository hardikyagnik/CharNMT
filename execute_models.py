import os
import subprocess
from subprocess import Popen, PIPE, STDOUT

EPOCHS = 100
DIR_OUTPUT = "output"
if not os.path.exists(DIR_OUTPUT):
    os.makedirs(DIR_OUTPUT)

def train_models(epochs, no_attention=False):
    if no_attention:
        cmd = f"python3 attention_nmt.py --nepochs {epochs} --no_attention --cuda --checkpoint_dir checkpoints/no-attn/py3"
        FILE_NAME = 'result_without_attention.txt'
    else:
        cmd = f"python3 attention_nmt.py --nepochs {epochs} --cuda --checkpoint_dir checkpoints/attn/py3"
        FILE_NAME = 'result_with_attention.txt'
    
    proc = Popen(cmd.split(' '), stdout=PIPE, stderr=PIPE)
    output, error = proc.communicate()
    print(f"OUTPUT\n\n{output}")
    print(f"ERROR\n\n{error}")
    f = open(os.path.join(DIR_OUTPUT, FILE_NAME), 'wb')
    f.write(output)
    f.close()
    

train_models(EPOCHS, no_attention=False)
train_models(EPOCHS, no_attention=True)
