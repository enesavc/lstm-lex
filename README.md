# LSTM-Lex
This repo host the necessay files, data and codes/scripts that were utilized for the LSTM lexicon project.

## Paper Citation
Avcu, E., Hwang, M., Brown, K., Gow, D. (underreview). A Tale of Two Lexica: Investigating Computational Pressures on Word Representation with Deep Neural Networks. Submitted to Neurobiology of Language Special Issue: Cognitive Computational Neuroscience of Language.

## Data
### Training Data
883 Words.txt has all the words we have used in this project. These words were based on a set of 260 phonetically diverse monomorphemic English words. We then used 20 of the most commonly used English affixes (15 suffixes and 5 prefixes) to generate inflected forms of these words. We used the Apple text-to-speech program Say to generate pronunciations (audio) for all the words in our lexicon. We used 10 different speakers (five females and five males) to ensure a diverse set of tokens for each word (each word has 10 tokens, making a total of 8830 total training items). The mean utterance duration was 684 ms (range: 335–1130 ms).
