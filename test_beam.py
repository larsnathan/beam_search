import fire
import json
import os
import numpy as np
import tensorflow as tf
import time
import beam_search

import model, sample, encoder

sess = None
logits = None
context = None
g_batch_size = None

def get_out_logits(con):
    return sess.run(logits, feed_dict={
                                context: [con for _ in range(g_batch_size)]
                            })


def test_beam(
    model_name='124M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=10,
    temperature=1, # .5 usually has numbered steps, .7 usually does not
    beam_width=3,
    top_k=None,
    top_p=1,
    models_dir='models',
    input_samples=[],
):
    models_dir = os.path.expanduser(os.path.expandvars(models_dir))
    global g_batch_size
    if batch_size is None:
        g_batch_size = 1
    assert nsamples % batch_size == 0
    g_batch_size = batch_size

    top_k = beam_width #Set the top_k to the beam_width to only find the beam_width number of logits

    enc = encoder.get_encoder(model_name, models_dir)
    hparams = model.default_hparams()
    with open(os.path.join(models_dir, model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    global sess
    with tf.Session(graph=tf.Graph()) as sess:
        global context
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        global logits
        logits = sample.get_logits(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        input_iter = 0
        raw_text = ""
        if (input_iter < len(input_samples)):
            raw_text = input_samples[input_iter]
            input_iter += 1
            print(raw_text)
        elif (len(input_samples) == 0):
            raw_text = input("Model prompt >>> ")

        while not raw_text:
            print('Prompt should not be empty!')
            raw_text = input("Model prompt >>> ")
        context_tokens = enc.encode(raw_text)

        out_contexts = beam_search.beam_search(get_out_logits, context_tokens)

        print("Beam Search:")
        for con in out_contexts:
            print(enc.decode(con))

        out_contexts = beam_search.efn_search(get_out_logits, context_tokens)

        print("\nEFN Search:")
        for con in out_contexts:
            print(enc.decode(con))


if __name__ == '__main__':
    fire.Fire(test_beam)