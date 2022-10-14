import os
import re
import numpy as np


HEADER = ".+\\*\\*\\*.*?START.*?GUTENBERG.*?\\*\\*\\*"
FOOTER = "\\*\\*\\*.*?END.*?GUTENBERG.*?\\*\\*\\*.+"


def gutenberg_strip(raw_text):
    """ Normalize the raw text of a Project Gutenberg ebook. """

    assert re.search(HEADER, raw_text, re.DOTALL)
    assert re.search(FOOTER, raw_text, re.DOTALL)
    body = "%s(.*)%s" % (HEADER, FOOTER)
    match = re.match(body, raw_text, re.DOTALL)
    assert match is not None
    stripped = match.group()

    single_linebreak_pattern = "\n(?!\n)"
    unwrapped = re.sub(single_linebreak_pattern, " ", stripped)

    return unwrapped


def load_corpus(textdir):
    """ Create a single np.uint8 array from all texts in the corpus. """

    filepaths = [os.path.join(textdir, fn) for fn in
                 os.listdir(textdir) if fn.endswith(".txt")]
    
    if not filepaths:
        raise Exception("You appear to have no data in `texts/`")

    print("Loading books . . .")
    integer_sequences = []
    for fp in filepaths:
        print("Loading %r . . ." % fp, end=" ", flush=True)
        with open(fp, encoding="ISO-8859-1") as source:
            edited = gutenberg_strip(source.read())
            indices = [ord(c) for c in edited + "\n\n"]
            if not all(i <= 256 for i in indices):
                bigs = sorted(chr(i) for i in set(indices) if i >= 256)
                raise Exception("Unsupported characters: %s" % bigs)
            intseq = np.uint8(indices)
        print("Done; contains %s characters." % len(intseq))
        integer_sequences.append(intseq)
    print("Finished loading %s books.\n" % len(integer_sequences))

    return np.concatenate(integer_sequences)


def convert_to_snippets(long_sequence, snippet_length, shuffle=False):
    """ Create a stack of consecutive snippets from a single array. """

    length = len(long_sequence)
    divisible_length = length - (length % snippet_length)
    truncated_sequence = long_sequence[:divisible_length]
    snippets = np.reshape(truncated_sequence, [-1, snippet_length])

    if shuffle:
        reordering = np.random.permutation(len(snippets))
        snippets = snippets[reordering, :]
    
    return snippets


if __name__ == "__main__":

    corpus = load_corpus("texts")

    print("The corpus consists of %.1f million characters."
          % (len(corpus) / 1e6))