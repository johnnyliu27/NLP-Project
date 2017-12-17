# NLP-Project
Final project for 6.806/6.864 on identifying similar questions on stackexchange.

This project was written in Python 3.6 with PyTorch 0.1.12_2.

Files too large to push to github:

AskUbuntu vectors and corpus files. They can be found at https://github.com/taolei87/askubuntu  
Pruned embeddings: https://www.dropbox.com/s/flop5p7z4k58u3u/glove.combined.300d.txt.gz?dl=0  
Direct Transfer LSTM: https://www.dropbox.com/s/34biw5pyds370q5/lstm_direct?dl=0  

The iPython notebooks detail the code we used to train/evaluate models. They are currently "frozen" in the state we used to arrive at our best models. In order to evaluate our best models, simply change the code which loads a model corresponding to an epoch to instead read from the corresponding best model, e.g. "cnn_model" for Testing.ipynb, our cnn model trainer.