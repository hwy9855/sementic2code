import numpy as np
from sklearn.cluster import MiniBatchKMeans
from tqdm import tqdm
import pickle as pk
import random
import os


def load_reps(reps_path):
    batches = len(os.listdir(reps_path))
    data_paths = [str(i) for i in range(batches)]
    return data_paths


if __name__ == '__main__':
    reps_path = '/disk/scratch/s2088661/bert_reps/'
    codes = 256
    batch_size = 1024
    epochs = 10

    kmeans = MiniBatchKMeans(n_clusters=codes,
                             random_state=0,
                             batch_size=batch_size)
    reps_paths = load_reps(reps_path)

    print('Start training mini-batch kmeans.')
    for epoch in range(epochs):
        reps_paths_ = reps_paths.copy()
        random.shuffle(reps_paths_)
        reps = []
        cnt = 0
        for path in tqdm(reps_paths_):
            with open(reps_path + path, 'rb') as f:
                cnt += 1
                reps.append(pk.load(f))
            if cnt >= 8:
                cnt = 0
                reps_np = np.vstack(reps)
                reps = []
                kmeans.partial_fit(reps_np)

    print('Start generating codes.')
    with open('bert.codes', 'w') as f_out:
        for path in tqdm(reps_paths):
            with open(reps_path + path, 'rb') as f:
                reps = pk.load(f)
            bert_codes = kmeans.predict(reps)
            for code in bert_codes:
                f_out.write(str(code))
                f_out.write('\n')

