code_path = 'ft.codes'
tgt_in_path = '../mydata/wmt14.en.sp.filtered'
tgt_out_path = 'ft.tgt'

codes = []
with open(code_path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        codes.append(int(line.split()[0]))

i = 0
with open(tgt_in_path, 'r') as f_in:
    with open(tgt_out_path, 'w') as f_out:
        lines = f_in.readlines()
        for line in lines:
            if line == '\n':
                f_out.write('\n')
            else:
                tmp = '<c' + str(codes[i]+1) + '> <eoc> ' + line
                f_out.write(tmp)
            i += 1

