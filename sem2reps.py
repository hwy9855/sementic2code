from load_data import load_data
from sem_model import *
from scipy.cluster.vq import kmeans, whiten
from tqdm import tqdm
import numpy as np
import argparse
import math
import pickle as pk

def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str)
    parser.add_argument('--data_path', type=str,
                        default='../mydata/wmt14.en.sp.filtered')
    parser.add_argument('--bits', type=int,
                        default=8)
    parser.add_argument('--batch_size', type=int,
                        default=1024)
    parser.add_argument('--save_path', type=str,
                        default='ft_reps/')
    return parser.parse_args()


def save_reps_batch(batch, reps, save_path):
    with open(save_path + str(batch), 'wb') as f:
        pk.dump(reps, f)


if __name__ == '__main__':
    args = parse_arg()
    data = load_data(args.data_path)
    codes = 2 ** args.bits
    code_res = []

    if args.mode == 'random':
        generate_random_code(codes)

    else:
        if args.mode == 'bert':
            model = BERT2rep('bert-base-uncased')
            print('Semantic model is BERT.')
            print('Generating semantic representations...')

            batches = math.ceil(len(data) / args.batch_size)
            for batch in tqdm(range(batches)):
                left = batch * args.batch_size
                right = min((batch + 1) * args.batch_size, len(data))
                reps = model.generate_rep(data[left:right]).detach().cpu().numpy()
                save_reps_batch(batch, reps, args.save_path)

        elif args.mode == 'ft':
            model = fastText2rep('cc.en.300.bin')
            print('Semantic model is Fasttext.')
            print('Generating semantic representations...')

            batches = math.ceil(len(data) / args.batch_size)
            for batch in tqdm(range(batches)):
                reps = []
                left = batch * args.batch_size
                right = min((batch + 1) * args.batch_size, len(data))
                texts = data[left:right]
                for text in texts:
                    reps.append(model.generate_rep(text))
                reps = np.vstack(reps)
                print(reps.shape)
                save_reps_batch(batch, reps, args.save_path)
