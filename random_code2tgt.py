import numpy as np

tgt_in_path = '../mydata/wmt14.en.sp.filtered'
tgt_out_path = 'random.tgt'

np.random.seed(42)

with open(tgt_in_path, 'r') as f_in:
    with open(tgt_out_path, 'w') as f_out:
        lines = f_in.readlines()
        for line in lines:
            if line == '\n':
                f_out.write('\n')
            else:
                code = np.random.randint(256)
                tmp = '<c' + str(code + 1) + '> <eoc> ' + line
                f_out.write(tmp)
