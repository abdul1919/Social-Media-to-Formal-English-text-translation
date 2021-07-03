# Importing libraries
#import streamlit as st 
from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, LSTM, Dense
from keras.models import load_model

import flask
app = Flask(__name__)

# Importing data
df = pd.read_csv('df.csv')

# Splitting data into train and test

train, validation = train_test_split(df, test_size=0.1,random_state=0)

# for one sentence we will be adding <end> token so that the tokanizer learns the word <end>
# with this we can use only one tokenizer for both encoder output and decoder output
train.iloc[0]['english_inp']= str(train.iloc[0]['english_inp'])+' <end>'
train.iloc[0]['english_out']= str(train.iloc[0]['english_out'])+' <end>'

tknizer_sms = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tknizer_sms.fit_on_texts(train['sms'].values)
tknizer_eng = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
tknizer_eng.fit_on_texts(train['english_inp'].values)

vocab_size_eng=len(tknizer_eng.word_index.keys())
vocab_size_sms=len(tknizer_sms.word_index.keys())

class Dataset:
    def __init__(self, data, tknizer_ita, tknizer_eng, max_len):
        self.encoder_inps = data['sms'].values
        self.decoder_inps = data['english_inp'].values
        self.decoder_outs = data['english_out'].values
        self.tknizer_eng = tknizer_eng
        self.tknizer_sms = tknizer_sms
        self.max_len = max_len

    def __getitem__(self, i):
        self.encoder_seq = self.tknizer_sms.texts_to_sequences([self.encoder_inps[i]]) # need to pass list of values
        self.decoder_inp_seq = self.tknizer_eng.texts_to_sequences([self.decoder_inps[i]])
        self.decoder_out_seq = self.tknizer_eng.texts_to_sequences([self.decoder_outs[i]])

        self.encoder_seq = pad_sequences(self.encoder_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_inp_seq = pad_sequences(self.decoder_inp_seq, maxlen=self.max_len, dtype='int32', padding='post')
        self.decoder_out_seq = pad_sequences(self.decoder_out_seq, maxlen=self.max_len, dtype='int32', padding='post')
        return self.encoder_seq, self.decoder_inp_seq, self.decoder_out_seq

    def __len__(self): # your model.fit_gen requires this function
        return len(self.encoder_inps)

    
class Dataloder(tf.keras.utils.Sequence):    
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.dataset.encoder_inps))


    def __getitem__(self, i):
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        batch = [np.squeeze(np.stack(samples, axis=1), axis=0) for samples in zip(*data)]
        # we are creating data like ([italian, english_inp], english_out) these are already converted into seq
        return tuple([[batch[0],batch[1]],batch[2]])

    def __len__(self):  # your model.fit_gen requires this function
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        self.indexes = np.random.permutation(self.indexes)
        

# Creating sample data to call the model once in order to load weights that was saved earlier
train_sample = train.iloc[:1]

train_dataset_sample = Dataset(train_sample, tknizer_sms, tknizer_eng, 20)

train_dataloader_sample = Dataloder(train_dataset_sample, batch_size=1)

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size,lstm_size, input_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.input_length = input_length
        self.lstm_size= lstm_size
        self.lstm_output = 0
        self.lstm_state_h=0
        self.lstm_state_c=0
        
    def build(self, input_shape):
        self.embedding = Embedding(input_dim=self.vocab_size+1, output_dim=self.embedding_size, input_length=self.input_length,
                           mask_zero=True, name="embedding_layer_encoder")
        self.lstm = LSTM(self.lstm_size, return_state=True, return_sequences=True, name="Encoder_LSTM")
        
    def call(self, input_sentances, training=True):
        input_embedd                           = self.embedding(input_sentances)
        self.lstm_output, self.lstm_state_h,self.lstm_state_c = self.lstm(input_embedd)
        return self.lstm_output, self.lstm_state_h,self.lstm_state_c
    def initialize_states(self,batch_size):
        return self.lstm_state_h,self.lstm_state_c
    
class Attention(tf.keras.layers.Layer):
    def __init__(self,scoring_function, att_units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(att_units)
        self.W2 = tf.keras.layers.Dense(att_units)
        self.V = tf.keras.layers.Dense(1)
        self.scoring_function = scoring_function

        if self.scoring_function=='dot':
            pass
  
    def call(self,decoder_hidden_state,encoder_output):
    
        if self.scoring_function == 'dot':
            # Implement Dot score function here
            decoder_hidden_state = tf.keras.layers.Reshape((decoder_hidden_state.shape[1], 1))(decoder_hidden_state)
            score = tf.keras.layers.dot([encoder_output, decoder_hidden_state],[2, 1])
            attention_weights = tf.nn.softmax(score, axis=1)
            context_vector = attention_weights * encoder_output
            context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector,attention_weights
    
class One_Step_Decoder(tf.keras.Model):
    def __init__(self,tar_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
        super().__init__()
        self.dec_units = dec_units
        self.input_length = input_length
        self.embedding = tf.keras.layers.Embedding(input_dim=tar_vocab_size+1, output_dim=embedding_dim, input_length=input_length)
        self.score_fun = score_fun
        self.att_units = att_units
        self.lstm = tf.keras.layers.LSTM(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(tar_vocab_size)
        self.attention_layer = Attention(self.score_fun,self.att_units)
        # Initialize decoder embedding layer, LSTM and any other objects needed

    def call(self,input_to_decoder, encoder_output, state_h,state_c):
    
        embed_out = self.embedding(input_to_decoder)
    
        context_vec , Att_weights = self.attention_layer(state_h, encoder_output)

        Concate = tf.concat([tf.expand_dims(context_vec, 1), embed_out], axis=-1)

        decoder_out, state_h1, state_c1 = self.lstm(Concate)

        decoder_out = tf.reshape(decoder_out, (-1, decoder_out.shape[2]))
        output = self.fc(decoder_out)

        return output,state_h1,state_c1,Att_weights,context_vec
    
class Decoder(tf.keras.Model):
    def __init__(self,out_vocab_size, embedding_dim, input_length, dec_units ,score_fun ,att_units):
      
        super(Decoder, self).__init__()
        self.out_vocab_size = out_vocab_size
        self.embedding_dim = embedding_dim
        self.input_length = input_length
        self.dec_units = dec_units
        self.score_fun = score_fun
        self.att_units = att_units
        self.onestepdecoder = One_Step_Decoder(self.out_vocab_size, self.embedding_dim, self.input_length, self.dec_units ,self.score_fun ,self.att_units)

    def call(self, input_to_decoder,encoder_output,decoder_hidden_state,decoder_cell_state ):

        all_outputs = tf.TensorArray(tf.float32,size=tf.shape(input_to_decoder)[1],name='output_arrays')
        for i in range(tf.shape(input_to_decoder)[1]):
            output,state_h1,state_c1,Att_weights,context_vec = self.onestepdecoder(input_to_decoder[:,i:i+1],encoder_output,decoder_hidden_state,decoder_cell_state)
            all_outputs = all_outputs.write(i,output)

        all_outputs = tf.transpose(all_outputs.stack(),[1,0,2])

        return all_outputs   
    
class Attention_model(Model):
    def __init__(self, encoder_inputs_length,decoder_inputs_length, output_vocab_size,scoring_fun,units):
        super(Attention_model,self).__init__() # https://stackoverflow.com/a/27134600/4084039
        self.score = scoring_fun
        self.encoder = Encoder(vocab_size=vocab_size_sms+1, embedding_size=100, input_length=encoder_inputs_length, lstm_size=512)
        self.decoder = Decoder(out_vocab_size=vocab_size_eng+1,embedding_dim=100,input_length=decoder_inputs_length,dec_units=units,score_fun=self.score,att_units=units)        
        
    def call(self, data):
        input,output = data[0], data[1]
        print(input.shape,output.shape)
        initial_state                                           = self.encoder.initialize_states(1024)
        encoder_output, encoder_h, encoder_c                    = self.encoder(input,initial_state)
        decoder_h_state , decoder_c_state                       = encoder_h, encoder_c
        decoder_output                                          = self.decoder(output,encoder_output,  decoder_h_state , decoder_c_state)
        return decoder_output
    
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction='none')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

model_2  = Attention_model(encoder_inputs_length=20,decoder_inputs_length=20,output_vocab_size=vocab_size_eng,scoring_fun='dot',units=32)
optimizer = tf.keras.optimizers.Adam()

model_2.compile(optimizer=optimizer,loss=loss_function)
model_2.fit_generator(train_dataloader_sample)

model_2.load_weights('my_model_weights_new.h5')

#def welcome():
#    return "Welcome All"

@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/index')
def index():
    return flask.render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    to_predict_list = request.form.to_dict()
    input_sentence = to_predict_list["SMS_text"]
    ### Tokenizing
    tok_result = tknizer_sms.texts_to_sequences([input_sentence])
    pad_result = pad_sequences(tok_result, maxlen=20, padding='post')

    ### Encoder
    en_outputs,state_h , state_c = model_2.layers[0](tf.constant(pad_result))
    onestepdecoder=One_Step_Decoder(vocab_size_eng, 100, 20, 32 ,'dot' ,32)
    cur_vec = tf.constant([[tknizer_eng.word_index['<start>']]])
    pred = [] ## To save predicted

    for i in range(20):
        predictions = model_2.layers[1](cur_vec,en_outputs, state_h, state_c)
        _,_,_,attention_weights,_=onestepdecoder(cur_vec,en_outputs, state_h, state_c)

        cur_vec = np.reshape(np.argmax(predictions), (1, 1))
        pred.append(tknizer_eng.index_word[cur_vec[0][0]])
    
        if(pred[-1]=='<end>'):
            break
        translated_sentence = ' '.join(pred)

    return jsonify({'prediction': translated_sentence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)