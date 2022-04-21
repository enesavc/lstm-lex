# LSTM-Lex
This repo host the necessay files, data and codes/scripts that were utilized for the LSTM lexicon project.

## Summary


## Paper Citation
Avcu, E., Hwang, M., Brown, K., Gow, D. (underreview). A Tale of Two Lexica: Investigating Computational Pressures on Word Representation with Deep Neural Networks. Submitted to Neurobiology of Language Special Issue: Cognitive Computational Neuroscience of Language.

## Hardware, and Software
Simulations were conducted on a Linux workstation with an Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz, 98-gb of RAM, and an NVIDIA Quadro RTX 8000 (48-gb) graphics card. Simulations were conducted using Python 3.6, TensorFlow 2.2.0, and Keras 2.4.3. Each model requires approximately 48 hours to train on this workstation. This repository provides an up-to-date container with all necessary explanations and jupyter notebooks for running our training code and analyses.


## Data
### Training Data
Data/883 Words.txt has all the words we have used in this project. These words were based on a set of 260 phonetically diverse monomorphemic English words. We then used 20 of the most commonly used English affixes (15 suffixes and 5 prefixes) to generate inflected forms of these words. We used the Apple text-to-speech program Say to generate pronunciations (audio) for all the words in our lexicon. We used 10 different speakers (five females and five males) to ensure a diverse set of tokens for each word (each word has 10 tokens, making a total of 8830 total training items). The mean utterance duration was 684 ms (range: 335–1130 ms).

### Cochleagrams
We used cochleagrams of each sound file as the input to the network. A cochleagram is a spectrotemporal representation of an auditory signal designed to capture cochlear frequency decomposition (i.e., it has overlapping spectral filters whose width increases with center frequency). The cochleagrams were created using code that produced cochleagrams in other studies (Feather et. al., 2019; Kell et al., 2018). See below figure for a schematic representation of audio to a cochleagram. Cochleagram generation was done in Python, using the numpy and scipy libraries (Harris et al., 2020; Oliphant, 2007), with signal trimming via librosa (McFee et al., 2015). Please refer to https://github.com/jenellefeather/tfcochleagram for cochleagram generation. Data/Coch_Gen.ipynb shows how we created cochleagrams.

![image](https://user-images.githubusercontent.com/32641692/164516870-9198cd2c-5a5b-47e8-a102-030ecf4c1da8.png)

## Models
We created two separate LSTM models and trained them independently on the same training data (8830 tokens for 883 words). Dorsal network was trained to differentiate between words using arbitrary information and a ventral network was trained to distinguish words based on distributional properties. See the paper for details. Models/Dorsal.ipynb and Models/Ventral.ipynb have the model training codes with necessay componenets (partition data and output labels). This folder also has best dorsal and ventral models which were used to calculate accuracy and extract activation patterns. See model structure below.

![image](https://user-images.githubusercontent.com/32641692/164526923-b5879933-edd6-4482-89cc-3bdfc01f92c5.png)

During the training, we checkpointed (saved the model and weights in a checkpoint file) the model every 100 epochs, so that we could load the model and calculate our accuracy metrics as training time increased. We used cosine similarity as the accuracy metric. See the paper for details.

## Generalization Tasks
We created 3 tasks. The first task was the word identification task where each word token was grouped into its own word type (with ten tokens) resulting in 883 categories. The second task was the articulatory generalization task where words were grouped into seven categories based on the manner of articulation of the initial (onset) phoneme: vowels, voiced stops, voiceless stops, fricatives, nasals, liquids, or glides. The final task was the semantic/syntactic generalization task where words were grouped into nine parts of speech categories: singular (NN) and plural (NNS) nouns, adjectives (JJ) and comparative adjectives (JJR), base (VB), past (VBD), gerund (VBG), present (VBZ) verbs and finally adverbs (RB). We extracted the activation in the hidden layer of the best performing network (before the sigmoid layer where the classification happens) to 8830 words (cochleagrams) categorized into 883 (word identification task), 7 (articulatory task), and 9 (semantic/syntactic task) classes, respectively. The features (hidden layer activations) were extracted from both models at every time point (0 to 225). See the paper for the details of decoding steps which were done in Python using the numpy and sklearn libraries (Pedrogosa et al., 2011). See generalization task results below, which shows representations learned for one task do not support the other which means task-specific representations are required for each task

![image](https://user-images.githubusercontent.com/32641692/164541042-173bdc70-88da-4ed8-a492-c0a6f13d2ae7.png)

