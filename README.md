# SlavicGPT
Building, training, and fine-tuning of medium-sized GPTs on Russian text and Slavic literature.


### Initial experiments

`dev.ipynb` contains my initial experiments with the bigram model, a simple baseline language model, as well as my exploration into developing a more powerful model that can capture more complex dependencies between tokens using self-attention.

Then, in `gpt.py`, I took what I learned to build a character-level, transformer-decoder language model based on OpenAI's GPT architecture. I trained it on Russian text obtained from various genres and authors in the classic Russian literary canon—poetry, prose, and publicism from 12 different authors including Tolstoy, Dostoevsky, and Pushkin. I wrote `retrieve_russian_text.py` to process this set of literary works into one text file and clean it to remove garbage characters and sequences. I used my resulting `data/tiny-russian-lit/very_clean_tiny_russian_lit.txt` file as the dataset for this first experiment. This dataset has ~34.8 million characters (for a character-level model, this is also the total number of tokens) and a vocabulary size of 87.

My GPT for this experiment had ~10.8 million parameters.

Given a newline character as the starting context, the untrained model generated the following sample:
```
Т?лШ:д&;Нбл-—уЁпЮчЧ;кМлИыАлЦІ:аьлЫрЫгбБдлчІЖЫБдБ—хХы?дАЪЉ!ы?кГЮ́д
І!ДДХ
ъуіИЬшев шИЩл!ш,йдАЁкдеФЩЛІѝщ
вжчкЬОП;ІвКогыЙъ́о&ЙюдЕЦ—КЮ.И,ш.Ъ—ныАЕЯч̀бЦмцЗ–ЉцзЪiєЦрА?шєфАХэыюХзрЧшИЮв.х
С?̀Я,-іiщi!аЯєИЙАЉГДЗЗж&е’ЖдЕЗх &ЗІЉъъЉП—ЖнукИЙе́Хи’ыёАшлУЗ ЮєСжБТлУ.ЗЮ
```
Gibberish!

After training for 15 minutes on one A100 GPU and getting down to a validation loss of ~1.48, the model generated the following sample:
```
Как ли едет грубопытство, сделал этим правом на два днереже помок, должить в небо!
Она наведливо одну и поохожала несколько в глаз, а ее мило особенно в кошки-с.
Старик шгнул мартвы в церковь мой и дадущий и сердца разволял его. Дона и не видовала бы
```
This is starting to look a lot more like Russian, but it is still pretty nonsensical if you actually read (or try to translate) it. Nevertheless, this initial model has clearly learned some level of patterns and dependencies in the Russian language and how tokens 'communicate' across a sequence.