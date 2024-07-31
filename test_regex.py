import re

EXCLUDE = r'_|`|\||~|§|\*|\[\d+\]|\{\d+\}|#\d+|#'
text = "§ hello_ | I am| {123} + [9] with * and ~~~ ~ and ~ w§here§ || # #444"
clean_text = re.sub(EXCLUDE, '', text)
print(clean_text)


text2 = "This is a test / string with // multiple / and // slashes//// / hello//world//hi."

# Replace '//' with '. '
text2 = re.sub(r'(?<!/)//(?!/)', '. ', text2)

# Replace '/' with ''
text2 = re.sub(r'(?<!/)/(?!/)', '', text2)

print(text2)
