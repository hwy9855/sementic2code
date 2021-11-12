from tqdm import tqdm


def load_data(data_path):
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('Loading data...')
        for line in tqdm(lines):
            words = []
            tmp = line.split()
            for word in tmp:
                if word[0] == '▁':
                    words.append(word[1:])
                else:
                    words[-1] += word
            data.append(' '.join(words))
    return data