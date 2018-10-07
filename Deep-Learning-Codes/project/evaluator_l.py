import unicodedata
import string
import re
import random
import time
import math
import io
import pickle
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from mymodel import EncoderRNN
from mymodel import AttnDecoderRNN
from mymodel import Attn
from language_translator import Lang

USE_CUDA = True
LANG1 = 'en'
LANG2 = 'es'
MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
#teacher_forcing_ratio = 0.5
#clip = 5.0

#attn_model = 'general'
#hidden_size = 500
#n_layers = 2
#dropout_p = 0.05




# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def filter_pair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH #and   p[1].startswith(good_prefixes)

def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]

def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))

    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))

    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])

    return input_lang, output_lang, pairs


# Return a list of indexes, one for each word in the sentence
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
#     print('var =', var)
    if USE_CUDA: var = var.cuda()
    return var

def variables_from_pair(pair, input_lang, output_lang, pairs ):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)

def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
#    lines = io.open('data/%s-%s.txt' % (lang1, lang2)).read().strip().split('\n')
    lang1_lines = io.open('data/%s_%s/%s.txt' %(LANG1, LANG2, LANG1), encoding='utf8').read().strip().split('\n')
    lang2_lines = io.open('data/%s_%s/%s.txt' %(LANG1, LANG2, LANG2), encoding='utf8').read().strip().split('\n')
#    
#print(lang1_lines) #['Je ne supporte pas ce type.', 'Je ne supporte pas ce type.', 'Pour une fois dans ma vie je fais un bon geste... Et ça ne sert à rien.',
    
    
    # Split every line into pairs and normalize
#    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    pairs = [[normalize_string(s) for s in l] for l in zip(lang1_lines, lang2_lines)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def evaluate(sentence, input_lang, output_lang, pairs, encoder, decoder, max_length=MAX_LENGTH):
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    
    # Run through encoder
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        decoder_context = decoder_context.cuda()

    decoder_hidden = encoder_hidden
    
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    
    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
            
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
        if USE_CUDA: decoder_input = decoder_input.cuda()
    
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]

def evaluate_randomly(encoder, decoder):
    input_lang, output_lang, pairs = prepare_data(LANG1, LANG2, True)
    for i in range(5):
        pair = random.choice(pairs)
        output_words, attentions = evaluate(pair[0], input_lang, output_lang, pairs, encoder, decoder)
        output_sentence = ' '.join(output_words)
    
        print('>', pair[0])
        print('=', pair[1])
        print('<', output_sentence)
        print('')
        show_attention(pair[0], output_words, attentions)

def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    fig.savefig('attention/'+LANG1+'_'+LANG2+'_'+input_sentence +'.png')   # save the figure to file
    plt.close(fig) 
 #   plt.close()

#def evaluate_and_show_attention(input_sentence):
#    input_lang, output_lang, pairs = prepare_data(LANG1, LANG2, True)
#    pair = random.choice(pairs)
#    output_words, decoder_attn = evaluate(pair[0], input_lang, output_lang, pairs, encoder, decoder)
#    output_words, attentions = evaluate(input_sentence)
#    print('input =', input_sentence)
#    print('output =', ' '.join(output_words))
#    show_attention(input_sentence, output_words, attentions)
def main():    
# LOAD the models

    encoder = torch.load('model/%s_%s_encoder'%(LANG1, LANG2)+ '.pt')
    decoder = torch.load('model/%s_%s_decoder'%(LANG1, LANG2)+ '.pt')
    with open('losses_'+LANG1+'_'+LANG2+'.txt', 'rb') as fp:
        plot_losses = pickle.load(fp)
    evaluate_randomly(encoder, decoder)

if __name__=="__main__":
    main()

