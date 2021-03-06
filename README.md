# LSTM-Lex
This repo host the necessay files, data and codes/scripts that were utilized for the LSTM lexicon project.

## Paper Citation
Avcu, E., Hwang, M., Brown, K., Gow, D. (under review). A tale of two lexica: Investigating computational pressures on word representation with deep neural networks. Submitted to Neurobiology of Language Special Issue: Cognitive Computational Neuroscience of Language.

## Summary of The Project
Words play a pivotal role in almost every aspect of language processing. The dual-stream model of spoken language processing (Hickok & Poeppel, 2007) suggests that processing is organized broadly into parallel dorsal and ventral processing streams concerned with dissociable aspects of motor and conceptual-semantic processing. Drawing on converging evidence from pathology, neuroimaging, behavioral research, and histology, Gow (2012) proposes that each pathway has its own lexicon or lexical interface area, which mediates mappings between acoustic-phonetic representation and stream-specific processing. The purpose of this project is to ask why humans have evolved two lexicons, rather than a single lexicon that interacts with both processing streams. Specifically, we ask whether computational demands on the mapping between acoustic-phonetic input and stream-specific processing create pressure for the development of different computationally efficient featural representations of words in the dorsal and ventral streams. We find that networks trained on the mapping between sound and articulation perform poorly in recognizing the mapping between sound and meaning and vice versa. We then show networks developed internal representations reflecting specialized task optimized functions without explicit training. Together, these findings indicate that functional specialization of word representation may reflect a computational optimization given the structure of the tasks that brains must solve, namely, different featural projections of wordform may be needed to support efficient linguistic processing.

## Hardware, and Software
Simulations were conducted on a Linux workstation with an Intel(R) Xeon(R) Gold 5218 CPU @ 2.30GHz, 98-gb of RAM, and an NVIDIA Quadro RTX 8000 (48-gb) graphics card. Simulations were conducted using Python 3.6, TensorFlow 2.2.0, and Keras 2.4.3. Each model requires approximately 48 hours to train on this workstation. This repository provides an up-to-date container with all necessary explanations and jupyter notebooks for running our training code and analyses.

## Data
### Training Data
Data/883 Words.txt has all the words we have used in this project. These words were based on a set of 260 phonetically diverse monomorphemic English words. We then used 20 of the most commonly used English affixes (15 suffixes and 5 prefixes) to generate inflected forms of these words. We used the Apple text-to-speech program Say to generate pronunciations (audio) for all the words in our lexicon. We used 10 different speakers (five females and five males) to ensure a diverse set of tokens for each word (each word has 10 tokens, making a total of 8830 total training items). The mean utterance duration was 684 ms (range: 335???1130 ms).

### Cochleagrams
We used cochleagrams of each sound file as the input to the network. A cochleagram is a spectrotemporal representation of an auditory signal designed to capture cochlear frequency decomposition (i.e., it has overlapping spectral filters whose width increases with center frequency). The cochleagrams were created using code that produced cochleagrams in other studies (Feather et. al., 2019; Kell et al., 2018). See below figure for a schematic representation of audio to a cochleagram. Cochleagram generation was done in Python, using the numpy and scipy libraries (Harris et al., 2020; Oliphant, 2007), with signal trimming via librosa (McFee et al., 2015). Please refer to https://github.com/jenellefeather/tfcochleagram for the details and required libraries for cochleagram generation. Data/Coch_Gen.ipynb shows how we created cochleagrams.
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/32641692/164516870-9198cd2c-5a5b-47e8-a102-030ecf4c1da8.png"
  >
</p>

## Models
We created two separate LSTM models and trained them independently on the same training data (8830 tokens for 883 words). Dorsal network was trained to differentiate between words using arbitrary information and a ventral network was trained to distinguish words based on distributional properties. See the paper for details. Models/Dorsal.ipynb and Models/Ventral.ipynb files have the model training codes with necessay componenets (partition data and output labels). Models folder also has best dorsal and ventral models which were used to calculate accuracy and extract activation patterns. See model structure below.
<p align="center">
  <img 
    src="https://user-images.githubusercontent.com/32641692/165430692-d73f5a77-91a1-41d8-b8d8-19568281d369.png"
  >
</p>

During the training, we checkpointed (saved the model and weights in a checkpoint file) the model every 100 epochs, so that we could load the model and calculate our accuracy metrics as training time increased. We used cosine similarity as the accuracy metric. Please refer to the paper for details.

## Generalization Tasks
We created 3 tasks. The first task was the word identification task where each word token was grouped into its own word type (with ten tokens) resulting in 883 categories. The second task was the articulatory generalization task where words were grouped into seven categories based on the manner of articulation of the initial (onset) phoneme: vowels, voiced stops, voiceless stops, fricatives, nasals, liquids, or glides. The final task was the semantic/syntactic generalization task where words were grouped into nine parts of speech categories: singular (NN) and plural (NNS) nouns, adjectives (JJ) and comparative adjectives (JJR), base (VB), past (VBD), gerund (VBG), present (VBZ) verbs and finally adverbs (RB). We extracted the activation in the hidden layer of the best performing network (before the sigmoid layer where the classification happens) to 8830 words (cochleagrams) categorized into 883 (word identification task), 7 (articulatory task), and 9 (semantic/syntactic task) classes, respectively. The features (hidden layer activations) were extracted from both models at every time point (0 to 225). See the paper for the details of decoding steps which were done in Python using the numpy and sklearn libraries (Pedrogosa et al., 2011). See generalization task results below, which shows representations learned for one task do not support the other which means task-specific representations are required for each task
<p align="center">
  <img 
    width="800"
    height="400"
    src="https://user-images.githubusercontent.com/32641692/164541042-173bdc70-88da-4ed8-a492-c0a6f13d2ae7.png"
  >
</p>

## Hidden Unit Selectivity Analyses
We checked whether hidden units of both networks encode information related to phoneme, morpheme, and root representation. Therefore, we developed three selectivity indices (SIs). The Phonemic Selectivity Index (PSI), adapted from Mesgarani et al. (2014) and Magnuson et al. (2020), quantifies the hidden unit???s response to a target phoneme relative to all the other phonemes. The Morpheme Selectivity Index (MSI) quantifies the selectivity of each hidden unit???s response to a target derivational or inflectional morpheme relative to all the other morphemes. We used all the root plus one affix words in our lexicon to extract each hidden unit???s response to each of the 20 morphemes over the full-time window. The Root Selectivity Index (RSI) shows the hidden unit???s response to a target root where all the other roots are controlled. We used 40 randomly chosen roots among the 252 roots in our lexicon to extract each hidden unit???s response to these roots over the full-time window. See below for an example figure that shows hidden unit selectivities to 39 English phonemes. Hidden Analysis folder has all the necessary codes/scripts to create a figure like below.
![image](https://user-images.githubusercontent.com/32641692/164761802-9d7391e2-9372-4b26-930f-3cdaee468f10.png)


