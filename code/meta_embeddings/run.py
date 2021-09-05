import argparse
from model import CAEME, AAEME, SED
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler, Dataset
import gensim
from tqdm import tqdm
import numpy as np
import codecs
import os
import math
from transformers import get_linear_schedule_with_warmup
import sys
from sklearn.model_selection import train_test_split


from sklearn.preprocessing import normalize
from scipy.sparse.linalg import svds, eigs

sys.path.append('/hdd_var/mtikhomi/work_folder/simlex/simlex_git')
from evaluate import simlex_analysis

try:
    sys.path.append('/hdd_var/mtikhomi/work_folder/projects/taxonomy_enrichment_project/code')
    sys.path.append('/hdd_var/mtikhomi/work_folder/simlex/simlex_git')
    from evaluate import simlex_analysis
    #from taxonomy_code.models import RuWordNet, EnWordNet
except:
    pass

from taxonomy_code.models import RuWordNet, EnWordNet, RuThes

lang = ''
def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    return word_vectors

def print_simlex(wv_dict):
    eval_path = '/hdd_var/mtikhomi/work_folder/simlex/simlex_git/evaluation'
    if lang == 'ru':
        language = 'russian'
        wordsim_source = 'russe-similarity'
    elif lang == 'en':
        language = 'english'
        wordsim_source = language
    c1, cov1 = simlex_analysis(wv_dict, language, eval_path=eval_path)
    print(f"SimLex-999 score and coverage: {c1} {cov1}")

    c2, cov2 = simlex_analysis(wv_dict, language, source=wordsim_source, eval_path=eval_path)
    print(f"WordSim Russe Similarity score and coverage: {c2} {cov2}")

def save_vectors(wv, wv_path):
    with codecs.open(wv_path, 'w', 'utf-8') as file_descr:
        wv_size = wv[list(wv.keys())[0]].shape[0]

        nwords = len({w: wv[w] for w in wv if wv[w].shape[0] == wv_size})
        print(nwords)
        file_descr.write(f'{nwords} {wv_size}')
        for w in tqdm(wv):
            if wv[w].shape[0] != wv_size:
                continue
            vector = ' '.join([str(val) for val in wv[w]])
            file_descr.write(f'\n{w[3:]} {vector}')

def find_closes_w_vec(w, word_vectors_list, wv_index):
    most_sim = {}
    if f'_name{w}' in word_vectors_list[wv_index]:
        return word_vectors_list[wv_index][f'_name{w}']

    return np.zeros(word_vectors_list[wv_index].vector_size, dtype=np.float32)

    for i, wv in enumerate(word_vectors_list):
        if w not in wv:
            continue
        for ms_w, cos in wv.most_similar(w, topn=10):
            if ms_w not in word_vectors_list[wv_index]:
                continue
            if ms_w not in most_sim:
                most_sim[ms_w] = cos
            else:
                most_sim[ms_w] += cos
    most_sim = sorted([(ms_w, cos) for ms_w, cos in most_sim.items()], key=lambda x: -x[1])
    if len(most_sim) == 0:
        return np.zeros(word_vectors_list[wv_index].vector_size, dtype=np.float32)
        
    return word_vectors_list[wv_index][most_sim[0][0]]

def get_cooc_vectors(word_vectors_list):
    total_vocab = {}
    min_w_count = 1#len(word_vectors_list)#int(len(word_vectors_list) / 2)
    for wv in word_vectors_list:
        for w in wv.vocab:
            if w not in total_vocab:
                total_vocab[w] = 0
            total_vocab[w] += 1
    #vocab_oov = [w for w in total_vocab if total_vocab[w] == 1]
    union_vocab = sorted(list([w for w in total_vocab if total_vocab[w] == len(word_vectors_list)]))
    oov_vocab = sorted(list([w for w in total_vocab if total_vocab[w] < len(word_vectors_list) and total_vocab[w] >= min_w_count]))

    total_vocab = union_vocab + oov_vocab
    #total_vocab.update(vocab_oov[:1000])
    #total_vocab.update(competition_words)
#    total_vocab = sorted(list(total_vocab))
    matrixes = []
    for i in range(len(word_vectors_list)):
        matrixes.append([])
    matrix = []
    for w in tqdm(total_vocab):
        v_list = []
        for i, wv in enumerate(word_vectors_list):
            if w not in wv:
                v_list.append(find_closes_w_vec(w, word_vectors_list, i))
                matrixes[i].append(find_closes_w_vec(w, word_vectors_list, i))
                #print(matrixes[i][-1].dtype, matrixes[i][-1].sum())
            else:
                v_list.append(wv[w])
                matrixes[i].append(wv[w])
        #if stack_matrix:
        #    v_list = np.hstack(v_list)
            #matrixes[i].append(wv[w])
        matrix.append(np.array(v_list))
    #if stack_matrix:
        #matrix = np.vstack(matrix)
    return total_vocab, matrix, matrixes

def get_concat_vectors(vectors):
    vocab, _, matrixes = get_cooc_vectors(vectors)
    print(len(matrixes))
    print(len(matrixes[0]))
    conc_wv = np.concatenate(matrixes, axis=1)
    conc_wv_dict = {f'{lang}_{vocab[i]}': conc_wv[i] for i in range(len(vocab))}
    print('Concat')
    try:
        print_simlex(conc_wv_dict)
    except:
        pass

    return conc_wv_dict

def get_svd_vectors(vectors, vectors_size):
    vocab, _, matrixes = get_cooc_vectors(vectors)
    print(len(matrixes))
    print(len(matrixes[0]))
    conc_wv = np.concatenate(matrixes, axis=1)
    conc_wv_dict = {f'{lang}_{vocab[i]}': conc_wv[i] for i in range(len(vocab))}
    print('Concat')
    try:
        print_simlex(conc_wv_dict)
    except:
        pass
    conc_wv_svd, _, _ = svds(conc_wv, k=vectors_size)
    
    conc_wv_svd_dict = {f'{lang}_{vocab[i]}': conc_wv_svd[i] for i in range(len(vocab))}
    print('SVD')
    
    try:
        print_simlex(conc_wv_svd_dict)
    except:
        pass
   
    return conc_wv_svd_dict


class DatasetWithConctrains(Dataset):
    def __init__(self, matrix, constrains, targets):
        self.matrix = [torch.tensor(np.vstack(matrix[i])) for i in range(matrix.shape[0])]
        self.constrains = constrains
        self.targets = targets
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        inputs = [self.matrix[i][idx] for i in range(len(self.matrix))]
        constrains = self.constrains[idx]
        constrains = [self.matrix[i][constrains] for i in range(len(self.matrix))]
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)
        return inputs, constrains, targets

def prepare_data(wv_list):
    vocab, _, matrix, = get_cooc_vectors(wv_list)
    matrix = np.array(matrix)
    return vocab, matrix, TensorDataset(*[torch.tensor(np.vstack(matrix[i])).to(torch.device('cuda')) for i in range(len(wv_list))])

def prepare_data_thes_constrains(wv_list, thesaurus, max_pos_count, neg_count):
    vocab, _, matrix, = get_cooc_vectors(wv_list)
    matrix = np.array(matrix)

    word2id = {}
    for i, w in enumerate(vocab):
        word2id[w] = i

    sense2synid = {w: [] for w in thesaurus.senses}
    for synset_id in thesaurus.synsets:
        for sense in thesaurus.synsets[synset_id].synset_words:
            sense2synid[sense].append(synset_id)

    all_constrains = []
    all_targets = []
    for i, w in tqdm(enumerate(vocab)):
        constrains = [word2id[w]] * (max_pos_count + neg_count)
        targets = [1] * (max_pos_count + neg_count)
        if w in sense2synid:
            pos_synset_words = []
            for synid in sense2synid[w]:
                pos_synset_words += [synset_w for synset_w in thesaurus.synsets[synid].synset_words if synset_w != w and synset_w in word2id]
            neg_words = [vocab[np.random.randint(0, len(vocab))] for i in range(neg_count)]
            pos_words = np.random.choice(pos_synset_words, min(max_pos_count, len(pos_synset_words)))

            targets = [1] * len(pos_words) + [-1] * len(neg_words)
            targets += [1] * (max_pos_count + neg_count - len(targets))
            constrains = [word2id[w] for w in pos_words]
            constrains += [word2id[w] for w in neg_words]
            constrains += [word2id[w]] * (max_pos_count + neg_count - len(constrains))
            
        
        active_examples.append(sum(np.array(constrains) != 0) != 0)
        all_constrains.append(constrains)
        all_targets.append(targets)
    dataset = DatasetWithConctrains(matrix, all_constrains, all_targets)
    return vocab, matrix, dataset

def create_dataset(vocab, full_matrix, indices=None, thes_constrains=False, thesaurus=None, constrains_count=3):
    if not thes_constrains:
        if indices is not None:
            matrix = [torch.tensor(np.vstack(full_matrix[i][indices])) for i in range(len(full_matrix))]
        else:
            matrix = [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))]
        print([m.shape for m in matrix])
        return TensorDataset(*matrix)

    sense2synid = {w: [] for w in thesaurus.senses}
    for synset_id in thesaurus.synsets:
        for sense in thesaurus.synsets[synset_id].synset_words:
            sense2synid[sense].append(synset_id)

    if indices is None:
        indices = [i for i in range(len(vocab))]

    word2id = {}
    id2word = {}
    for i in indices:
        w = vocab[i]
        word2id[w] = i
        id2word[i] = w

    all_pos_constrains = []
    for i in tqdm(indices):
        w = id2word[i]
        pos_constrains = []
        if w in sense2synid:
            for synid in sense2synid[w]:
                pos_constrains += [word2id[synset_w] for synset_w in thesaurus.synsets[synid].synset_words if synset_w != w and synset_w in word2id]
            for hypo in thesaurus.synsets[synid].rels.get('hyponym', []):
                pos_constrains += [word2id[synset_w] for synset_w in hypo.synset_words if synset_w != w and synset_w in word2id]
            for hyper in thesaurus.synsets[synid].rels.get('hypernym', []):
                pos_constrains += [word2id[synset_w] for synset_w in hyper.synset_words if synset_w != w and synset_w in word2id]
                #for hyperhypo in hyper.rels.get('hyponym', []):
                #    pos_constrains += [word2id[synset_w] for synset_w in hyperhypo.synset_words if synset_w != w and synset_w in word2id]

        pos_constrains = list(set(pos_constrains))
        all_pos_constrains.append(pos_constrains)

    if indices is not None:
        matrix = [torch.tensor(np.vstack(full_matrix[i][indices])) for i in range(len(full_matrix))]
    else:
        matrix = [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))]
    return DatasetWithConctrainsOnline(vocab, [torch.tensor(np.vstack(full_matrix[i])) for i in range(len(full_matrix))], matrix, all_pos_constrains, constrains_count=constrains_count)

class DatasetWithConctrainsOnline(Dataset):
    def __init__(self, full_vocab, full_matrix, matrix, pos_constrains, constrains_count=3):
        self.full_vocab = full_vocab
        self.full_matrix = full_matrix
        self.matrix = matrix
        self.pos_constrains = pos_constrains
        self.constrains_count = constrains_count

    def __len__(self):
        return len(self.matrix[0])

    def __getitem__(self, idx):
        inputs = [self.matrix[i][idx] for i in range(len(self.matrix))]
        def normalize_vec(v):
            norm = np.linalg.norm(v)
            if norm == 0:
                return v
            return v / norm

        if  len(self.pos_constrains[idx]) > 0:
            pos_constrains = np.random.choice(self.pos_constrains[idx], self.constrains_count)
            pos_constrains = [self.full_matrix[i][pos_constrains] for i in range(len(self.full_matrix))]
        else:
            pos_constrains = [[] for i in range(len(inputs))]
            for i, m in enumerate(inputs):

                for j in range(self.constrains_count):
                    noised_vec = m + torch.tensor(normalize_vec(np.random.rand(m.shape[0]) - 0.5) / 10, dtype=torch.float32)
                    pos_constrains[i].append(noised_vec)

                pos_constrains[i] = torch.stack(pos_constrains[i])

        neg_constrains = [np.random.randint(0, len(self.full_vocab)) for i in range(self.constrains_count)]
        neg_constrains = [self.full_matrix[i][neg_constrains] for i in range(len(self.full_matrix))]


        return inputs, pos_constrains, neg_constrains

def eval_simlex(model, vocab):
    meta_dict = {}
    simlex_vocab = set()

    eval_path = '/hdd_var/mtikhomi/work_folder/simlex/simlex_git/evaluation/'

    if lang == 'ru':
        language = 'russian'
        wordsim_path = f'{eval_path}/ws-353-russe/hj-wordsim353-similarity.csv'
        wordsim_sep = ','
    elif lang == 'en':
        language = 'english'
        wordsim_path = f'{eval_path}/ws-353/wordsim353-{language}.txt'
        wordsim_sep = '\t'

    with codecs.open(f'{eval_path}/simlex-{language}.txt', 'r', 'utf-8') as file:
        for line in file:
            w1, w2 = line.split('\t')[:2]
            simlex_vocab.update([w1, w2])

    with codecs.open(wordsim_path, 'r', 'utf-8') as file:
        for line in file:
            w1, w2 = line.split(wordsim_sep)[:2]
            simlex_vocab.update([w1, w2])

    with torch.no_grad():
        for w in tqdm(simlex_vocab):
            if w not in vocab:
                continue
            i = vocab.index(w)
            wv_list = [torch.tensor(matrix[wv_i][i].reshape(1, matrix[wv_i][i].shape[0])).to(torch.device('cuda')) for wv_i in range(len(vectors))]
            meta = model.extract(wv_list)
            meta_dict[f'{lang}_{w}'] = meta[0].detach().cpu().numpy()

    meta_dict = normalise_word_vectors(meta_dict)
    print_simlex(meta_dict)

def eval(args, model, dev_dataloader, epoch):
    model.eval()
    epoch_losses = []
    for batch_idx, batch in tqdm(enumerate(dev_dataloader)):
        batch = [b for b in batch]
        if args.thes_constrains:
            inputs, pos_constrains, neg_constrains = batch
#                print(len(inputs), len(pos_constrains), len(neg_constrains))
#                print(inputs[0].shape, pos_constrains[0].shape, neg_constrains[0].shape)
        else:
            inputs = batch
        inputs = [d.to(torch.device('cuda')) for d in inputs]

        if args.thes_constrains:
            pos_constrains = [d.to(torch.device('cuda')) for d in pos_constrains]
            neg_constrains = [d.to(torch.device('cuda')) for d in neg_constrains]
        else:
            pos_constrains = None
            neg_constrains = None
        loss = model(inputs, pos_constrains, neg_constrains)

        if args.thes_constrains and len(loss) == 3:
            loss, loss_direct, loss_constrains = loss

            epoch_losses.append((float(loss.detach().cpu().numpy()), float(loss_direct.detach().cpu().numpy()), float(loss_constrains.detach().cpu().numpy())))
        else:
            epoch_losses.append(loss.detach().cpu().numpy())

    loss_info = np.mean(epoch_losses, axis=0)#, np.std(epoch_losses, axis=0)
    print(f'Dev loss after {epoch} epoch = {loss_info}')
    #print(epoch_losses[-10:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vectors', nargs='+')
    parser.add_argument('--model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--logging_every_epochs', default=10, type=int)
    parser.add_argument('--result_path')
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--wv_weights', nargs='+', default=None, type=float)
    parser.add_argument('--emb_dim', type=int)
    parser.add_argument('--thes_constrains', action='store_true')
    parser.add_argument('--thes_path', type=str)
    parser.add_argument('--constrains_count', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--dev_size', type=float, default=None)
    parser.add_argument('--lang', type=str, default='ru')
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--distance_type', type=str, default='mse')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--ruthes', action='store_true')
    args = parser.parse_args()

    lang = args.lang
    vectors = []
    for wv_path in args.vectors:
        try:
            vectors.append(gensim.models.KeyedVectors.load(wv_path))
        except:
            vectors.append(gensim.models.KeyedVectors.load_word2vec_format(wv_path, binary=False))

        vectors[-1].vectors = normalize(vectors[-1].vectors, axis=1)

    wv_shapes = [wv.vector_size for wv in vectors]
    if args.model == 'CAEME':
        model = CAEME(wv_shapes, args.wv_weights, alpha=args.alpha,
                      margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'AAEME':
        model = AAEME(wv_shapes, args.emb_dim, args.wv_weights,
                      alpha=args.alpha, margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'SED':
        model = SED(wv_shapes, args.emb_dim, args.wv_weights,
                    alpha=args.alpha, margin=args.margin, distance_type=args.distance_type)
    elif args.model == 'SVD':
        print(len(vectors))
        meta_dict = get_svd_vectors(vectors, args.emb_dim)
        save_vectors(meta_dict, args.result_path)
        exit()
    elif args.model == 'CONCAT':
        print(len(vectors))
        meta_dict = get_concat_vectors(vectors)
        save_vectors(meta_dict, args.result_path)
        exit()

    else:
        raise NotImplementedError

    model.to(torch.device('cuda'))
    print(model)

    thesaurus = None
    if args.thes_constrains:
        try:
            thesaurus = RuThes(args.thes_path) if args.ruthes else RuWordNet(args.thes_path)
        except:
            thesaurus = EnWordNet(args.thes_path)
    vocab, _, matrix, = get_cooc_vectors(vectors)
    matrix = np.array(matrix)

    print(matrix.shape)
    if args.dev_size is not None:
        assert args.dev_size > 0.0 and args.dev_size < 1.0
        train_indices, dev_indices = train_test_split(range(len(vocab)), test_size=args.dev_size, random_state=42)
        #print(set(train_indices).intersection(set(dev_indices)))
    else:
        train_indices = None
        dev_dataset = None

    print('Creating Train Dataset')
    train_dataset = create_dataset(vocab, matrix, train_indices, thes_constrains=args.thes_constrains, thesaurus=thesaurus, constrains_count=args.constrains_count)
    if args.dev_size is not None:
        print('Creating Dev Dataset')
        dev_dataset = create_dataset(vocab, matrix, dev_indices, thes_constrains=args.thes_constrains, thesaurus=thesaurus, constrains_count=args.constrains_count)

    #train_dataset[i]
    '''
    if args.thes_constrains:
        vocab, matrix, dataset = prepare_data_thes_constrains(vectors, thesaurus, args.max_pos_count, args.neg_count)
    else:
        vocab, matrix, dataset = prepare_data(vectors)
    '''

    sampler = RandomSampler(train_dataset)#SequentialSampler(dataset)
    loader = DataLoader(train_dataset, sampler=sampler, batch_size=1024, num_workers=args.num_workers)

    dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, sampler=dev_sampler, batch_size=1024, num_workers=args.num_workers)

    epochs = args.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=len(loader) * epochs
    )

    print('Start Training')
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        for batch_idx, batch in tqdm(enumerate(loader)):
            batch = [b for b in batch]
            if args.thes_constrains:
                inputs, pos_constrains, neg_constrains = batch
            else:
                inputs = batch
            inputs = [d.to(torch.device('cuda')) for d in inputs]
            
            if args.thes_constrains and epoch >=  epochs // 2:
                pos_constrains = [d.to(torch.device('cuda')) for d in pos_constrains]
                neg_constrains = [d.to(torch.device('cuda')) for d in neg_constrains]
            else:
                pos_constrains = None
                neg_constrains = None

            loss = model(inputs, pos_constrains, neg_constrains)

            if args.thes_constrains and epoch >= epochs // 2 and len(loss) == 3:
                loss, loss_direct, loss_constrains = loss

                epoch_losses.append((float(loss.detach().cpu().numpy()), float(loss_direct.detach().cpu().numpy()), float(loss_constrains.detach().cpu().numpy())))
            else:
                epoch_losses.append(loss.detach().cpu().numpy())

            loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step()

        #print([group['lr'] for group in optimizer.param_groups])
        #print(epoch_losses[-10:])
        print(np.mean(epoch_losses, axis=0))#, np.std(epoch_losses, axis=0))
        eval(args, model, dev_loader, epoch)
        if epoch % args.logging_every_epochs == 0:
            model.eval()
            try:
                eval_simlex(model, vocab)
            except Exception as e:
                print(f'Failed simlex: {e}')
            #eval(args, model, dev_loader, epoch)


    model.eval()
    eval(args, model, dev_loader, epoch)
    meta_dict = {}
    with torch.no_grad():
        for i, w in tqdm(enumerate(vocab)):
            wv_list = [torch.tensor(matrix[wv_i][i].reshape(1, matrix[wv_i][i].shape[0])).to(torch.device('cuda')) for wv_i in range(len(vectors))]
            meta = model.extract(wv_list)
            meta_dict[f'{lang}_{w}'] = meta[0].detach().cpu().numpy()

    meta_dict = normalise_word_vectors(meta_dict)

    try:
        print_simlex(meta_dict)
    except Exception as e:
        print(f'Failed simlex: {e}')

    save_vectors(meta_dict, args.result_path)

