# Character Level Neural Machine Translation
> Done as a part of acedamic project for `CS 583` - __Deep__ __Learning__   

Task: To train a model to transform English words to their transformed forms as per the rules bellow

### Transformation Rules
- If the first letter is a consonant, then that letter is moved to the end of the word and “ay” is appended, e.g., slow → lowsay
- If a word starts with a vowel, then append “way” at the end, e.g., amoeba → amoebaway
- Some consonant pairs like “sh” are moved together to end of the word with “ay” appended, e.g., shallow → allowshay

You can run train the model by executing the file `execute_models.py`.
You can see the results in the `output/` folder.
After each epoch of model training you can find the translation of the sentance `"the air conditioning is working"`.
