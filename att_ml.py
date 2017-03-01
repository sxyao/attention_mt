from collections import defaultdict,Counter

import numpy as np
import dynet as dy
# import _gdynet as dy
import random
import sys
import codecs
import nltk

class Attention:
    def __init__(self, model,training_src, training_tgt):
        self.model = model
        self.training_src, self.src_vocab, self.rsrc_vocab= self.change_word2id_genevoc(training_src)
        self.training_tgt, self.tgt_vocab, self.rtgt_vocab = self.change_word2id_genevoc_output(training_tgt)
        self.src_vocab_size = len(self.src_vocab)
        self.tgt_vocab_size = len(self.tgt_vocab)
        self.embed_size = 128
        self.src_lookup = model.add_lookup_parameters((self.src_vocab_size, self.embed_size))
        self.tgt_lookup = model.add_lookup_parameters((self.tgt_vocab_size, self.embed_size))
        self.hidden_size = 128
        self.layers = 1
        self.contextsize = self.hidden_size*2
        self.l2r_builder = dy.GRUBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.r2l_builder = dy.GRUBuilder(self.layers, self.embed_size, self.hidden_size, model)
        self.dec_builder = dy.GRUBuilder(self.layers, self.embed_size+self.contextsize, self.hidden_size*2, model)

        self.W_y = model.add_parameters((self.tgt_vocab_size, self.hidden_size*2+self.contextsize,))
        self.b_y = model.add_parameters(self.tgt_vocab_size)

        self.max_len = 50



    # Training step over a single sentence pair
    def step_batch(self, batch):
        dy.renew_cg()

        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)

        M_s = self.src_lookup
        M_t = self.tgt_lookup
        src_sent, tgt_sent = zip(*batch)
        src_sent = zip(*src_sent)
        tgt_sent = zip(*tgt_sent)
        src_sent_rev = list(reversed(src_sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()

        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            l2r_state = l2r_state.add_input(dy.lookup_batch(M_s,cw_l2r))
            r2l_state = r2l_state.add_input(dy.lookup_batch(M_s,cw_r2l))
            l2r_contexts.append(l2r_state.output())  # [<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output())  # [</S> x_n, x_{n-1}, ... <S>]

        # encoded_h1 = l2r_state.output()
        # tem1 = encoded_h1.npvalue()

        r2l_contexts.reverse()  # [<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))

        encoded_h = h_fs[-1]

        h_fs_matrix = dy.concatenate_cols(h_fs)
        h_fs_matrix_t = dy.transpose(h_fs_matrix)

        losses = []
        num_words = 0

        # Decoder
        c_t = dy.vecInput(self.hidden_size * 2)
        c_t.set([0 for i in xrange(self.contextsize)])
        encoded_h = dy.concatenate([encoded_h])
        dec_state = self.dec_builder.initial_state([encoded_h])
        for (cw, nw) in zip(tgt_sent[0:-1], tgt_sent[1:]):
            embed = dy.lookup_batch(M_t,cw)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            #calculate attention
            a_t = h_fs_matrix_t * h_e
            alignment = dy.softmax(a_t)
            c_t = h_fs_matrix * alignment
            ind_tem = dy.concatenate([h_e, c_t])
            ind_tem1 = W_y * ind_tem
            ind_tem2 = ind_tem1 + b_y
            loss = dy.pickneglogsoftmax_batch(ind_tem2, nw)  # to modify
            losses.append(loss)
            num_words += 1
        return dy.sum_batches(dy.esum(losses)), num_words

    def translate_sentence(self, sent):
        dy.renew_cg()
        W_y = dy.parameter(self.W_y)
        b_y = dy.parameter(self.b_y)
        M_s = self.src_lookup
        M_t = self.tgt_lookup

        src_sent = sent
        src_sent_rev = list(reversed(sent))

        # Bidirectional representations
        l2r_state = self.l2r_builder.initial_state()
        r2l_state = self.r2l_builder.initial_state()
        l2r_contexts = []
        r2l_contexts = []
        for (cw_l2r, cw_r2l) in zip(src_sent, src_sent_rev):
            l2r_state = l2r_state.add_input(M_s[cw_l2r])
            r2l_state = r2l_state.add_input(M_s[cw_r2l])
            l2r_contexts.append(l2r_state.output())  # [<S>, x_1, x_2, ..., </S>]
            r2l_contexts.append(r2l_state.output())  # [</S> x_n, x_{n-1}, ... <S>]
        r2l_contexts.reverse()  # [<S>, x_1, x_2, ..., </S>]

        # Combine the left and right representations for every word
        h_fs = []
        for (l2r_i, r2l_i) in zip(l2r_contexts, r2l_contexts):
            h_fs.append(dy.concatenate([l2r_i, r2l_i]))
        encoded_h = h_fs[-1]
        h_fs_matrix = dy.concatenate_cols(h_fs)
        h_fs_matrix_t = dy.transpose(h_fs_matrix)

        # Decoder
        trans_sentence = [u'<s>']
        cw = self.tgt_vocab[u'<s>']
        c_t = dy.vecInput(self.hidden_size * 2)
        c_t.set([0 for i in xrange(self.contextsize)])
        dec_state = self.dec_builder.initial_state([encoded_h])

        while len(trans_sentence) < self.max_len:
            embed = dy.lookup(M_t,cw)
            dec_state = dec_state.add_input(dy.concatenate([embed, c_t]))
            h_e = dec_state.output()
            # c_t = self.__attention_mlp(h_fs_matrix, h_e)

            # calculate attention
            a_t = h_fs_matrix_t * h_e
            alignment = dy.softmax(a_t)
            c_t = h_fs_matrix * alignment
            ind_tem = dy.concatenate([h_e, c_t])
            ind_tem1 = W_y * ind_tem
            ind_tem2 = ind_tem1 + b_y
            score = dy.softmax(ind_tem2)
            probs1 = score.npvalue()
            cw = np.argmax(probs1)
            if cw == self.tgt_vocab[u'</s>']:
                break
            trans_sentence.append(self.rtgt_vocab[cw])
        return trans_sentence[1:]

    def evaluate(self, dev_src,dev_tgt, outpath):
        hypos = []
        reference = []
        fout = codecs.open(outpath, 'w', 'utf-8')
        for i in xrange(len(dev_src)):
            sen_src = dev_src[i]
            p_sen = self.translate_sentence(sen_src)
            fout.write(u' '.join(p_sen) + u'\n')
            hypos.append(p_sen)
            reference.append([dev_tgt[i]])
        chencherry = nltk.translate.bleu_score.SmoothingFunction()
        fout.close()
        BLEU = nltk.translate.bleu_score.corpus_bleu(reference, hypos, smoothing_function=chencherry.method2)
        return BLEU*100

    def translate_corpus(self, src, outpath):
        fout = codecs.open(outpath,'w','utf-8')
        for sen_src in src:
            p_sen = self.translate_sentence(sen_src)
            fout.write(u' '.join(p_sen)+u'\n')
        fout.close()


    def change_word2id_genevoc(self,data):
        r_data = []
        vocab = defaultdict(lambda: len(vocab))
        vocab[u'<unk>'], vocab[u'<s>'], vocab[u'</s>']
        r_vocab = {0: u'<unk>', 1: u'<s>', 2: u'</s>'}
        for line in data:
            tem = []
            for word in line:
                tem.append(vocab[word])
                r_vocab[vocab[word]] = word
            r_data.append([vocab[u'<s>']]+tem+[vocab[u'</s>']])
        return [r_data,vocab,r_vocab]

    def change_word2id_genevoc_output(self,data):
        r_data = []
        counter = defaultdict(int)
        vocab = defaultdict(lambda: len(vocab))
        vocab[u'<unk>'], vocab[u'<s>'], vocab[u'</s>']
        r_vocab = {0: u'<unk>', 1: u'<s>', 2: u'</s>'}
        for line in data:
            for word in line:
                counter[word] += 1
                if counter[word]>=3:
                    vocab[word]
                    r_vocab[vocab[word]] = word
        r_data = self.change_word2id(data,vocab)
        return [r_data,vocab,r_vocab]

    def change_word2id(self, data, vocab):
        r_data = []
        for line in data:
            tem = []
            for word in line:
                if word in vocab:
                    tem.append(vocab[word])
                else:
                    tem.append(vocab[u'<unk>'])
            r_data.append([vocab[u'<s>']] + tem + [vocab[u'</s>']])
        return r_data

    # def change_id2word(self,data,r_vocab):
    #     r_data = []
    #     for line in data:
    #         tem = [r_vocab[word] for word in line]
    #         r_data.append(tem)
    #     return r_data

    def set_dropout(self, p):
        self.l2r_builder.set_dropout(p)
        self.r2l_builder.set_dropout(p)
        self.dec_builder.set_dropout(p)

    def disable_dropout(self):
        self.l2r_builder.disable_dropout()
        self.r2l_builder.disable_dropout()
        self.dec_builder.disable_dropout()


def read_file(filename):
    data = []
    for line in codecs.open(filename,'r','utf-8'):
        words = line[:-1].split(u' ')
        data.append(words)
    return data

def divide_batch(data,batch_size):
    lendict = defaultdict(list)
    for item in data:
        lendict[(len(item[0]), len(item[1]))].append(item)
    n_data = []
    for k in lendict:
        value = lendict[k]
        for ind in xrange(0, len(value), batch_size):
            n_data.append(value[ind:ind + batch_size])
    return n_data

def main(argv):
    model = dy.Model()
    trainer = dy.AdamTrainer(model)
    training_src = read_file(argv[1])
    training_tgt = read_file(argv[2])
    dev_src = read_file(argv[3])
    dev_tgt = read_file(argv[4])
    test_src = read_file(argv[5])
    test_tgt = read_file(argv[6])
    blind_src = read_file(argv[7])
    batch_size = 64
    attention = Attention(model, training_src, training_tgt)
    epoch_num = 40
    train_data = zip(attention.training_src, attention.training_tgt)

    dev_src = attention.change_word2id(dev_src, attention.src_vocab)
    test_src = attention.change_word2id(test_src, attention.src_vocab)
    blind_src = attention.change_word2id(blind_src, attention.src_vocab)

    train_data = divide_batch(train_data, batch_size)
    foutre = open('output3.txt', 'w')

    attention.disable_dropout()
    bleuscore = attention.evaluate(dev_src, dev_tgt, 'rnn_output/valid.primary.en')
    print 'Epoch 0', 'Valid', bleuscore
    foutre.write('Epoch ' + str(0) + ' Valid ' + str(bleuscore) + '\n')


    for i in xrange(epoch_num):
        print 'Epoch',i
        foutre.write('Epoch '+str(i)+':\n')
        np.random.shuffle(train_data)
        count = 1

        attention.set_dropout(0.5)
        for batch in train_data:
            losses, num_words = attention.step_batch(batch)
            if count % 5 == 0:
                tem = losses.value()
                print 'step', count, tem / (num_words * len(batch))
                foutre.write('step ' + str(count) + '  '+str(tem / (num_words * len(batch))) + '\n')
            losses.backward()
            trainer.update()
            if count == 800 and i==0:
                bleuscore = attention.evaluate(dev_src, dev_tgt, 'grudp_output/valid.primary.en' + str(i)+'_'+str(count))
                print 'Epoch', i, 'Valid', bleuscore
                foutre.write('Epoch ' + str(i) + ' Valid ' + str(bleuscore) + '\n')
            count += 1

        attention.disable_dropout()
        bleuscore = attention.evaluate(dev_src, dev_tgt, 'grudp_output/valid.primary.en' + str(i))
        print 'Epoch', i, 'Valid', bleuscore
        foutre.write('Epoch ' + str(i) + ' Valid ' + str(bleuscore) + '\n')

        if (i%5==0 and i>0) or i==0 or i==3:
            attention.model.save("grudp_model/m"+str(i), [attention.src_lookup, attention.tgt_lookup, attention.l2r_builder, attention.r2l_builder,attention.dec_builder,
                                                        attention.W_y,attention.b_y])
            test_bleu = attention.evaluate(test_src, test_tgt, 'grudp_output/test.primary.en'+str(i))
            print 'Epoch', i, 'Test', test_bleu
            foutre.write('Epoch ' + str(i) + ' Test ' + str(test_bleu) + '\n')
            attention.translate_corpus(blind_src,'grudp_output/blind.primary.en'+str(i))
    foutre.close()


if __name__ == '__main__':
    argv = ['','en-de/train.en-de.low.filt.de','en-de/train.en-de.low.filt.en','en-de/valid.en-de.low.de','en-de/valid.en-de.low.en','en-de/test.en-de.low.de',
            'en-de/test.en-de.low.en','en-de/blind.en-de.low.de']
    main(argv)
