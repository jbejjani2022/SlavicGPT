# SlavicGPT
Building, training, and fine-tuning of medium-sized GPTs on Russian text and Slavic literature.


### Initial experiments

`dev.ipynb` contains my initial experiments with the bigram model, a simple baseline language model, as well as my exploration into developing a more powerful model that can capture more complex dependencies between tokens via self-attention.

Then, in `gpt.py`, I took what I learned to build a character-level, transformer-decoder language model based on OpenAI's GPT architecture. I trained it on Russian text obtained from various genres and authors in the classic Russian literary canon—poetry, prose, and publicism from 12 different authors including Tolstoy, Dostoevsky, and Pushkin. I wrote `retrieve_russian_text.py` to process this set of literary works into one text file and clean it to remove garbage characters and sequences. I used my resulting `data/tiny-russian-lit/very_clean_tiny_russian_lit.txt` file as the dataset for this first experiment. This dataset has ~34.8 million characters (for a character-level model, this is also the total number of tokens) and a vocabulary size of 87.

My GPT for this experiment had ~10.8 million parameters.

Given a newline character as the starting context, the untrained model (with validation loss ~4.64) generated the following sample:
```
Т?лШ:д&;Нбл-—уЁпЮчЧ;кМлИыАлЦІ:аьлЫрЫгбБдлчІЖЫБдБ—хХы?дАЪЉ!ы?кГЮ́д
І!ДДХ
ъуіИЬшев шИЩл!ш,йдАЁкдеФЩЛІѝщ
вжчкЬОП;ІвКогыЙъ́о&ЙюдЕЦ—КЮ.И,ш.Ъ—ныАЕЯч̀бЦмцЗ–ЉцзЪiєЦрА?шєфАХэыюХзрЧшИЮв.х
С?̀Я,-іiщi!аЯєИЙАЉГДЗЗж&е’ЖдЕЗх &ЗІЉъъЉП—ЖнукИЙе́Хи’ыёАшлУЗ ЮєСжБТлУ.ЗЮ
```
Gibberish!

After training for ~15 minutes (comprising 12,000 training steps) on one A100 GPU, the model got down to a validation loss of ~1.40 and generated the following sample:
```
Кто-то сдал Катерин.
Ах, совсем не надобится Чичиков
На любовь!,
И, высказал ему обоим расставшиеся от этого, отдал отректы, слоймуны, поднимувшиеся знакомые вам,
на рубище, не об уставшей на бытую, но весело
изъявил ей.
Любовь, к своему важного учти
```
This is starting to look a lot more like Russian, but it is still pretty nonsensical if you actually read (or try to translate) it. Nevertheless, this initial model has clearly learned some level of patterns and dependencies in the Russian language and how tokens 'communicate' across a sequence.

### Acknowledgments

Many thanks to Andrej Karpathy for his amazing [tutorial](https://www.youtube.com/watch?v=kCc8FmEb1nY) and [nanoGPT](https://github.com/karpathy/nanoGPT) repo, which helped me learn what I needed for this project.