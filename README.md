# TayTech
Our goal was to create an End-To-End Text-to-Speech model that is capable of synthesizing Morgan Freeman's voice from text. We replicated the [Tacotron](https://arxiv.org/abs/1703.10135) model designed by Google. We were able to replicate the model. However, the lack of Morgan Freeman audio prevented us from generating his voice. Thus we switched to the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) to train our model.  

## Installation Instructions. 
1. Download the code. Download the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).
2. Download the [model](https://drive.google.com/drive/folders/1ch4Bus-b1kdShmdhEiUQ5ePYcM2azhZu). The model was too big to push to Github and was instead stored in a Google drive.
3. Modify these variables in `config.py`. 
   - `LOG_DIR`: location of model checkpoints
   - `MODEL_NAME`: the name of the model file to load
   - `DATA_PATH`: location of the dataset
   - `TEST_DATA`: if not passing in text to predict the audio output, the contents of this text file are used to synthesize audio
   - `SAVE_DIR`: where to save synthesized outputs
   - `DEVICE`: cpu or gpu depending on your computer
4. Modify the paths in `checkpoint` in the downloaded [model](https://drive.google.com/drive/folders/1ch4Bus-b1kdShmdhEiUQ5ePYcM2azhZu).
   
## Instruction to Run
1. Training - run `train.py` to train
2. Evaluation - run `eval.py` to evaluate
3. Synthesize - run `synthesizer.py` to synthesize output that is stored in `SAVE_DIR`. Either pass in text in the code or modify `TEST_DATA`

## Results
Although the code runs, we did not have suitable hardware to sufficiently train the model. Thus we were unable to synthesize new output. Output samples generated by the model in early stages of training are stored in the `Results` directory.

## Dependencies
Runs on Python 3
- Conda >= 4.4.10
- tensorflow >= 1.6
- librosa >= 0.6.0
- numpy >= 1.14.1
- matplotlib >= 2.2.2
- scipy >= 1.0.0
- tqdm

## Data
Two datasets were used. 
  - [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
  - [Morgan Freeman Audio](https://drive.google.com/drive/folders/1efzGhWzDOpSxnCnofrmSYX_j7cYokCzN)
  The model was trained on the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/). Although the Morgan Freeman Data was gathered, it was insufficient for training the model.

## References
- Tacotron: https://arxiv.org/abs/1703.10135
- Highway networks: https://arxiv.org/pdf/1505.00387.pdf
- LJ Speech Dataset: https://keithito.com/LJ-Speech-Dataset/
- Morgan Freeman Audio: https://youtube.com/
- Github repos: 
  - https://github.com/Kyubyong/tacotron
  - https://github.com/keithito/tacotron
