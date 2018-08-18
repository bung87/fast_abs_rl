
from os.path import join
import json
import argparse
from datetime import timedelta
from time import time
from collections import Counter, defaultdict
from itertools import product

from cytoolz import identity, concat, curry

from data.batcher import tokenize
from decoding import Abstractor, RLExtractor, DecodeDataset, BeamAbstractor
from decoding import make_html_safe
from torch import multiprocessing as mp
import torch
import operator as op
from functools import reduce

_PRUNE = defaultdict(
    lambda: 2,
    {1: 5, 2: 5, 3: 5, 4: 5, 5: 5, 6: 4, 7: 3, 8: 3}
)


def _compute_score(hyps):
    all_cnt = reduce(op.iadd, (h.gram_cnt for h in hyps), Counter())
    repeat = sum(c-1 for g, c in all_cnt.items() if c > 1)
    lp = sum(h.logprob for h in hyps) / sum(len(h.sequence) for h in hyps)
    return (-repeat, lp)


def _make_n_gram(sequence, n=2):
    return (tuple(sequence[i:i+n]) for i in range(len(sequence)-(n-1)))


def rerank_one(beams):
    @curry
    def process_beam(beam, n):
        for b in beam[:n]:
            b.gram_cnt = Counter(_make_n_gram(b.sequence))
        return beam[:n]
    beams = map(process_beam(n=_PRUNE[len(beams)]), beams)
    best_hyps = max(product(*beams), key=_compute_score)
    dec_outs = [h.sequence for h in best_hyps]
    return dec_outs


def rerank_mp(all_beams, ext_inds):
    beam_lists = [all_beams[i: i+n] for i, n in ext_inds if n > 0]
    with mp.Pool(8) as pool:
        reranked = pool.map(rerank_one, beam_lists)
    return list(concat(reranked))


def main(article_path,model_dir,  batch_size,
         beam_size, diverse, max_len, cuda):
    with open(join(model_dir, 'meta.json')) as f:
        meta = json.loads(f.read())
    if meta['net_args']['abstractor'] is None:
        # NOTE: if no abstractor is provided then
        #       the whole model would be extractive summarization
        assert beam_size == 1
        abstractor = identity
    else:
        if beam_size == 1:
            abstractor = Abstractor(join(model_dir, 'abstractor'),
                                    max_len, cuda)
        else:
            abstractor = BeamAbstractor(join(model_dir, 'abstractor'),
                                        max_len, cuda)
    extractor = RLExtractor(model_dir, cuda=cuda)
    with open(article_path) as f:
        raw_article_batch = f.readlines()
    tokenized_article_batch = map(tokenize(None), raw_article_batch)
    ext_arts = []
    ext_inds = []
    for raw_art_sents in tokenized_article_batch:
        print(raw_art_sents)
        ext = extractor(raw_art_sents)[:-1]  # exclude EOE
        if not ext:
            # use top-5 if nothing is extracted
            # in some rare cases rnn-ext does not extract at all
            ext = list(range(5))[:len(raw_art_sents)]
        else:
            ext = [i.item() for i in ext]
        ext_inds += [(len(ext_arts), len(ext))]
        ext_arts += [raw_art_sents[i] for i in ext]
    if beam_size > 1:
        all_beams = abstractor(ext_arts, beam_size, diverse)
        dec_outs = rerank_mp(all_beams, ext_inds)
    else:
        dec_outs = abstractor(ext_arts)
    # assert i == batch_size*i_debug
    for j, n in ext_inds:
        decoded_sents = [' '.join(dec) for dec in dec_outs[j:j+n]]
        print(decoded_sents)
        # with open(join(save_path, 'output/{}.dec'.format(i)),
        #             'w') as f:
        #     f.write(make_html_safe('\n'.join(decoded_sents)))
        # i += 1
        # print('{}/{} ({:.2f}%) decoded in {} seconds\r'.format(
        #     i, n_data, i/n_data*100,
        #     timedelta(seconds=int(time()-start))
        # ), end='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='run decoding of the full model (RL)')
    parser.add_argument('--path', required=True, help='path to text')
    parser.add_argument('--model_dir', help='root of the full model')

    # # dataset split
    # data = parser.add_mutually_exclusive_group(required=True)
    # data.add_argument('--val', action='store_true', help='use validation set')
    # data.add_argument('--test', action='store_true', help='use test set')

    # decode options
    parser.add_argument('--batch', type=int, action='store', default=32,
                        help='batch size of faster decoding')
    parser.add_argument('--beam', type=int, action='store', default=1,
                        help='beam size for beam-search (reranking included)')
    parser.add_argument('--div', type=float, action='store', default=1.0,
                        help='diverse ratio for the diverse beam-search')
    parser.add_argument('--max_dec_word', type=int, action='store', default=30,
                        help='maximun words to be decoded for the abstractor')

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable GPU training')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda

    # data_split = 'test' if args.test else 'val'
    main(args.path,args.model_dir,
         args.batch, args.beam, args.div,
         args.max_dec_word, args.cuda)
