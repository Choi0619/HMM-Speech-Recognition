# 🗣️ HMM Speech Recognition

## Description
This repository contains an implementation of a **Hidden Markov Model (HMM)** for **speech recognition** using the **Viterbi Algorithm**. The goal is to recognize speech by analyzing word sequences, calculating probabilities, and generating a confusion matrix to improve accuracy. This project was implemented using **Python**.  

![image](https://github.com/user-attachments/assets/ed92ec94-6245-4e12-a391-125745e29c08)

The steps include:
1. Reading data from `hmm.txt` and `dictionary.txt`.
2. Configuring word HMM and universal utterance HMM.
3. Implementing the Viterbi Algorithm to find the most probable word sequence.
4. Running `HResults.exe` to generate a confusion matrix and adjust errors for better recognition.

For more details, please refer to **hmm_Gyuhwan Choi.pdf**.

## 📁 File Structure:
- **HResults.exe**: Tool for generating the confusion matrix.
- **bigram.txt**: Bigram model for word probabilities.
- **unigram.txt**: Unigram model for word probabilities.
- **dictionary.txt**: Word dictionary for recognition.
- **calc.py**: Handles complex mathematical calculations.
- **hmm.py**: Implements HMM and Viterbi algorithm.
- **header.py**: Defines phone HMM structure and properties.
- **main.py**: Executes the Viterbi algorithm for speech recognition.
- **recognized.txt**: Output file with recognized words.
- **reference.txt**: Reference file for comparison.
- **vocabulary.txt**: List of words used in speech recognition.

## 🧠 How It Works:
- The program constructs an HMM by reading data from text files, configuring word and utterance models, and applying the **Viterbi Algorithm** to recognize word sequences.
- The process includes adjusting insertion and deletion errors to improve recognition accuracy.

## 🔍 Detailed Process:
1. **HMM Configuration**: Phone HMMs are built from the header and dictionary files.
2. **Viterbi Algorithm**: Calculates the most probable state sequences and converts them into word sequences.
3. **Result Analysis**: Generates a confusion matrix using `HResults.exe` to compare recognized words with reference data.

## 📄 Additional Information:
For a detailed explanation of the implementation, please refer to **hmm_Gyuhwan Choi.pdf**.

## Credits:
- Created by **Gyuhwan Choi**.
