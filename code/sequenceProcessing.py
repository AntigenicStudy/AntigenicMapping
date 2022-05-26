from mutation_h1 import *

np.random.seed(1)
random.seed(1)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Flu sequence analysis')
    parser.add_argument('model_name', type=str,
                        help='Type of language model (e.g., hmm, lstm)')
    parser.add_argument('--namespace', type=str, default='fluh1',
                        help='Model namespace')
    parser.add_argument('--dim', type=int, default=512,
                        help='Embedding dimension')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Training minibatch size')
    parser.add_argument('--n-epochs', type=int, default=14,
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Model checkpoint')
    parser.add_argument('--train', action='store_true',
                        help='Train model')
    parser.add_argument('--train-split', action='store_true',
                        help='Train model on portion of data')
    parser.add_argument('--test', action='store_true',
                        help='Test model')
    parser.add_argument('--embed', action='store_true',
                        help='Analyze embeddings')
    parser.add_argument('--semantics', action='store_true',
                        help='Analyze mutational semantic change')
    parser.add_argument('--combfit', action='store_true',
                        help='Analyze combinatorial fitness')
    args = parser.parse_args()
    return args

def load_meta(meta_fnames):
    metas = {}
    for fname in meta_fnames:
        with open(fname) as f:
            header = f.readline().rstrip().split('\t')
            for line in f:
                fields = line.rstrip().split('\t')
                accession = fields[1]
                meta = {}
                for key, value in zip(header, fields):
                    if key == 'Subtype':
                        meta[key] = value.strip('()').split('N')[0].split('/')[-1]
                    elif key == 'Collection Date':
                        meta[key] = int(value.split('/')[-1]) \
                                    if value != '-N/A-' else None
                    elif key == 'Host Species':
                        meta[key] = value.split(':')[1].split('/')[-1].lower()
                    else:
                        meta[key] = value
                metas[accession] = meta
    return metas

def process(fnames, meta_fnames):
    metas = load_meta(meta_fnames)

    seqs = {}
    for fname in fnames:
        for record in SeqIO.parse(fname, 'fasta'):
            if 'Reference_Perth2009_HA_coding_sequence' in record.description:
                continue
            if str(record.seq).count('X') > 10:
                continue
            if record.seq not in seqs:
                seqs[record.seq] = []
            #accession = record.description.split('|')[0].split(':')[1]
            accession = record.description.split('/')[2]
            seqs[record.seq].append(metas[accession])
    return seqs

def split_seqs(seqs, split_method='random'):
    train_seqs, test_seqs, val_seqs = {}, {}, {}

    old_cutoff = 1990
    new_cutoff = 2018

    tprint('Splitting seqs...')
    for seq in seqs:
        # Pick validation set based on date.
        seq_dates = [
            meta['Collection Date'] for meta in seqs[seq]
            if meta['Collection Date'] is not None
        ]
        if len(seq_dates) > 0:
            oldest_date = sorted(seq_dates)[0]
            if oldest_date < old_cutoff or oldest_date >= new_cutoff:
                test_seqs[seq] = seqs[seq]
                continue
        train_seqs[seq] = seqs[seq]
    tprint('{} train seqs, {} test seqs.'
           .format(len(train_seqs), len(test_seqs)))

    return train_seqs, test_seqs

def setup(args):
    fnames = [ '/xx/data/influenza/allh1.fasta' ] # change to your traning data
    meta_fnames = [ '/xx/data/influenza/allh1.tsv' ]# change to your metainfo file

    seqs = process(fnames, meta_fnames)

    seq_len = max([ len(seq) for seq in seqs ]) + 2
    vocab_size = len(AAs) + 2

    model = get_model(args, seq_len, vocab_size)

    return model, seqs

    args = parse_args()

    AAs = [
        'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
        'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
        'Y', 'V', 'X', 'Z', 'J', 'U', 'B', 'Z'
    ]
    vocabulary = { aa: idx + 1 for idx, aa in enumerate(sorted(AAs)) }

    model, seqs = setup(args)

    if 'esm' in args.model_name:
        args.checkpoint = args.model_name
    elif args.checkpoint is not None:
        model.model_.load_weights(args.checkpoint)
        tprint('Model summary:')
        tprint(model.model_.summary())

    if args.train or args.train_split or args.test:
        train_test(args, model, seqs, vocabulary, split_seqs)

    if args.embed:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')
        no_embed = { 'hmm' }
        if args.model_name in no_embed:
            raise ValueError('Embeddings not available for models: {}'
                             .format(', '.join(no_embed)))
        analyze_embedding(args, model, seqs, vocabulary)

    if args.semantics:
        if args.checkpoint is None and not args.train:
            raise ValueError('Model must be trained or loaded '
                             'from checkpoint.')

        from escape import load_doud2018, load_lee2019

        tprint('Lee et al. 2018...')
        seq_to_mutate, escape_seqs = load_doud2018()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          prob_cutoff=0., beta=1., plot_acquisition=True,
                          plot_namespace='flu_h1')
        tprint('')

        tprint('Lee et al. 2019...')
        seq_to_mutate, escape_seqs = load_lee2019()
        analyze_semantics(args, model, vocabulary, seq_to_mutate, escape_seqs,
                          prob_cutoff=0., beta=1., plot_acquisition=True,
                          plot_namespace='flu_h3')

    if args.combfit:
        from combinatorial_fitness import load_doud2016
        tprint('Doud et al. 2016...')
        wt_seqs, seqs_fitness = load_doud2016()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 prob_cutoff=0., beta=1.)

        from combinatorial_fitness import load_wu2020
        tprint('Wu et al. 2020...')
        wt_seqs, seqs_fitness = load_wu2020()
        strains = sorted(wt_seqs.keys())
        for strain in strains:
            analyze_comb_fitness(args, model, vocabulary,
                                 strain, wt_seqs[strain], seqs_fitness,
                                 prob_cutoff=0., beta=1.)
