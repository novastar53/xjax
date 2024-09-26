import logging

import jax
import jax.numpy as jnp

import pytest
from pytest import approx
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

import xjax
from xjax.signals import train_epoch_completed

# Module logger
logger = logging.getLogger(__name__)

def test_lr_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create LR model
    model = xjax.models.sklearn.logistic_regression()

    # I train model
    model = xjax.models.sklearn.train(model, X=X_train, y=y_train)

    # I test model
    y_score = xjax.models.sklearn.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)


def test_xgb_circle(rng: jax.Array):
    #
    # Givens
    #

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create xgb model
    model = xjax.models.sklearn.xgb_classifier()

    # I train model
    model = xjax.models.sklearn.train(model, X=X_train, y=y_train)

    # I test model
    y_score = xjax.models.sklearn.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get high score
    assert auroc > 0.9


def test_flax_2x1_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Event collectors
    losses = []

    #
    # Whens
    #

    # I create 2x1 flax model
    rng, model_rng = jax.random.split(rng)
    model, params = xjax.models.flax.mlp(rng=model_rng, inputs=2, outputs=1)

    # I subscribe to epoch completed signal for model
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        losses.append(loss)
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")

    # I train model
    params = xjax.models.flax.train(
        model,
        params=params,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.flax.predict(model, params=params, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)

    # We should have collected 1 loss for each epoch
    assert len(losses) == epochs


def test_flax_2x5x1_circle(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.001

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng, n=100000)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create 2x5x1 flax model
    rng, model_rng = jax.random.split(rng)
    model, params = xjax.models.flax.mlp(rng=model_rng, inputs=2, hiddens=[5], outputs=1)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")

    # I train model
    params = xjax.models.flax.train(
        model,
        params=params,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.flax.predict(model, params=params, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get "good" score
    assert auroc > 0.8


def test_torch_2x1_diagonal(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.01

    # I created diagonal dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.diagonal(rng=dataset_rng)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Event collectors
    losses = []

    #
    # Whens
    #

    # I create 2x1 torch model
    model = xjax.models.torch.mlp(inputs=2, outputs=1)

    # I subscribe to epoch completed signal for model
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        losses.append(loss)
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")

    # I train model
    model = xjax.models.torch.train(
        model,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.torch.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get perfect score
    assert auroc == approx(1.0, abs=0.001)

    # We should have collected 1 loss for each epoch
    assert len(losses) == epochs


def test_torch_2x5x1_circle(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    batch_size = 10
    epochs = 10
    learning_rate = 0.001

    # I created circle dataset
    rng, dataset_rng = jax.random.split(rng)
    X, y = xjax.datasets.circle(rng=dataset_rng, n=100000)

    # I split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    #
    # Whens
    #

    # I create 2x5x1 torch model
    model = xjax.models.torch.mlp(inputs=2, hiddens=[5], outputs=1)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")

    # I train model
    model = xjax.models.torch.train(
        model,
        X=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    # I test model
    y_score = xjax.models.torch.predict(model, X=X_test)
    auroc = roc_auc_score(y_test, y_score)

    #
    # Thens
    #

    # Model should get "good" score
    assert auroc > 0.8


@pytest.mark.nlp
def test_nlp_jax_sgns(rng: jax.Array):
    #
    # Givens
    #

    # Hyperparams
    embedding_size = 2
    batch_size = 10
<<<<<<< HEAD
    neg_per_pos = 20
    num_epochs = 3
    window_size = 3
=======
    neg_per_pos = 5
    num_epochs = 5
    window_size = 2
>>>>>>> upstream/main
    vocab_size = 5
    K=1

    # I generate a frequent word-pair sentence dataset
    sentences = xjax.datasets.freq_word_pair()

    # 
    # Whens
    #

    # I preprocess the sentences
    from collections import Counter

    # I count the tokens
    tokens = []
    for sentence in sentences:
        for tok in sentence:
            tokens.append(tok)

    # I generate a vocab
    counts = Counter(tokens).items()
    sorted_counts = sorted(counts, key=lambda k: k[1], reverse=True)
    vocab = sorted_counts[:vocab_size]

    # I generate a token lookup
    idx_to_tok = dict()
    tok_to_idx = dict()
    for idx,(tok, count) in enumerate(vocab):
        idx_to_tok[idx] = tok
        tok_to_idx[tok] = idx
    
    # I generate positive examples
    len_buffer = window_size//2

    def gen_pos_examples():
        dataset = []
        for s in sentences:
            for i in range(len(s)):
                for j in range(max(0,i-len_buffer), min(len(s),i+len_buffer)):
                    if i != j:
                        if s[i] in tok_to_idx and s[j] in tok_to_idx:
                            idx_i = tok_to_idx[s[i]]
                            idx_j = tok_to_idx[s[j]]
                            dataset.append((idx_i,idx_j))
        return dataset
        

    pos_examples = jnp.array(gen_pos_examples())


    # I create a skipgram model
    model, params = xjax.models.sgns.sgns(rng=rng, vocab_size=vocab_size, 
                                 embedding_size=embedding_size)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")
    
    # I train the model
    params = xjax.models.sgns.train(model, rng=rng, params=params,
                                   X=pos_examples,
                                   neg_per_pos=neg_per_pos,
                                   K= 1,
                                   epochs=num_epochs,
                                   batch_size=batch_size,
                                   learning_rate=0.01)


    #
    # Thens
    #


    # The cosine similarity between the two most frequently adjacent words should be higher
    # than that between two words that are never adjacent in the dataset
    
    def similarity_score(word1, word2):

        idx1 = tok_to_idx[word1]
        idx2 = tok_to_idx[word2]

        emb1 = params[idx1, :, 0]
        emb2 = params[idx2, :, 0]

        return jnp.dot(emb1,emb2)/(jnp.linalg.norm(emb1)*jnp.linalg.norm(emb2))

    assert similarity_score("apple", "banana") > similarity_score("peach", "grape")



@pytest.mark.nlp
def test_nlp_jax_char_rnn(rng: jax.Array):
    #
    # Givens
    #

    # I have a test dataset consisting of dinosaur names 
    txt = open("tests/datasets/dinos.txt", "r").read()

    # I clean the data
    txt = txt.lower()
    lines = [ l.strip() for l in txt.split("\n") ]

    # I create token lookup dictionaries
    counts = {}
    for line in lines:
        chars = list(line)
        for c in chars:
            if c not in counts:
                counts[c] = 1
            else:
                counts[c] += 1

    vocab = sorted(list(counts.keys()))
    tok_to_idx = {}
    idx_to_tok = {}

    for i,c in enumerate(vocab):
        tok_to_idx[c] = i
        idx_to_tok[i] = c

    # I add the stop character
    tok_to_idx["\n"] = 26
    idx_to_tok[26] = "\n"

    # I generate the dataset 

    VOCAB_SIZE = len(tok_to_idx)

    X = []
    Y = []
    for line in lines:
        tokens = [ tok_to_idx[c] for c in line ]
        X.append(jnp.array([VOCAB_SIZE] + tokens))
        Y.append(jnp.array(tokens + [VOCAB_SIZE]))

    #
    # Whens
    #

    # I define hyperparameters
    HIDDEN_SIZE = 100
    learning_rate = 0.01
    max_grad = 20
    epochs = 100

    # I define a character-rnn model
    params, model = xjax.models.char_rnn.char_rnn(rng, 27, HIDDEN_SIZE)

    # I log events
    @train_epoch_completed.connect_via(model)
    def collect_events(_, *, epoch, loss, elapsed, **__):
        logger.info(f"epoch={epoch}, loss={loss:0.4f}, elapsed={elapsed:0.4f}")

    # I train a character RNN model on the data 
    trained_params = xjax.models.char_rnn.train(model, rng=rng, params=params, X_train=X, Y_train=Y,
                                                 vocab_size=VOCAB_SIZE, epochs=epochs,
                                                   learning_rate=learning_rate,
                                                   max_grad=max_grad)

    # I generate sequences from a prefix
    prefix_str = "p"
    prefix = [tok_to_idx[c] for c in prefix_str]
    generated = []
    for i in range(20):
        rng, sub_rng = jax.random.split(rng)
        y = xjax.models.char_rnn.generate(rng=sub_rng, prefix= prefix, params=trained_params, hidden_size=HIDDEN_SIZE, vocab_size=VOCAB_SIZE) 
        generated.append(y)

    # 
    # Thens 
    #

    # The predictions should have a 'high' probability of ending with 'us'
    num_correct = 0
    for g in generated:
        if "".join([idx_to_tok[i] for i in g[-3:]]) == "us\n":
            num_correct += 1

    assert(num_correct >= 2)


@pytest.mark.nlp
@pytest.mark.gru
def test_nlp_jax_gru(rng: jax.Array):

    params, model = xjax.models.gru.gru(rng, 10, 20)

    assert(len(params) == 11)
