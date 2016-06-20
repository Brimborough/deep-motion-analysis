# Theano Code for Motion Analysis


## Documentation

In this section, we describe how to use our code.
### Table of Contents
  * [LSTM Guide](#lstmg)
   * [Parameters](#lstmparams)
   * [Coding Decisions](#lstmdec)
   * [Classification](#lstmclass)
  
### LSTM Guide <a id="lstmg"></a>
<p> In this part I (<a href="https://github.com/joggino">Joe</a>) will describe how to use the LSTM and additional 
layers associated with the use of a LSTM.
</p>
#### Parameters <a id="lstmparams"></a>


#### Coding Decisions <a id="lstmdec"></a>
1. Not implemented offset labels, reference <a href="#lstmclass">Classification</a>

#### Classification <a id="lstmclass"></a>
To make this a classifier you must have the addition of a mean pooling layer and final activation layer.
Also I have not implemented a masked loss function, for which the labels don't occur at everytime step, for example only predicting the label for time step 1 after seeing 2 time steps. To see how to implement and use reference <a href="https://github.com/JonathanRaiman/theano_lstm#maskedloss-usage">here</a>.


