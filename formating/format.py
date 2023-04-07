# 0 incomplete sentence, 1 complete sentence
sentence_type = 0

with open("ToFormat.txt") as file:
    text = file.read()
    for line in text.splitlines():
        with open("Formatted.txt", "a") as file:
            file.write(f'("{line}", {sentence_type}),' )
            