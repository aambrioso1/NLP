"""
This follows Norvig's wonderful code found here on his GitHub page:

	https://github.com/norvig/pytudes/blob/master/ipynb/PropositionalLogic.ipynb

The goals to take English sentence and trun them into proposition logic statements.

The program uses regular expressions to match many of the patterns found in English sentences.

"""
import re

def Rule(output, *patterns):
    "A rule that produces `output` if the entire input matches any one of the `patterns`." 
    return (output, [name_group(pat) + '$' for pat in patterns])

def name_group(pat):
    "Replace '{Q}' with '(?P<Q>.+?)', which means 'match 1 or more characters, and call it Q'"
    return re.sub('{(.)}', r'(?P<\1>.+?)', pat)
            
def word(w):
    "Return a regex that matches w as a complete word (not letters inside a word)."
    return r'\b' + w + r'\b' # '\b' matches at word boundary

print(Rule('{P} ⇒ {Q}', 'if {P} then {Q}', 'if {P}, {Q}'))

rules = [
    Rule('{P} ⇒ {Q}',         'if {P} then {Q}', 'if {P}, {Q}'),
    Rule('{P} ⋁ {Q}',          'either {P} or else {Q}', 'either {P} or {Q}'),
    Rule('{P} ⋀ {Q}',          'both {P} and {Q}'),
    Rule('～{P} ⋀ ～{Q}',       'neither {P} nor {Q}'),
    Rule('～{A}{P} ⋀ ～{A}{Q}', '{A} neither {P} nor {Q}'), # The Kaiser neither ...
    Rule('～{Q} ⇒ {P}',        '{P} unless {Q}'),
    Rule('{P} ⇒ {Q}',          '{Q} provided that {P}', '{Q} whenever {P}', 
                               '{P} implies {Q}', '{P} therefore {Q}', 
                               '{Q}, if {P}', '{Q} if {P}', '{P} only if {Q}'),
    Rule('{P} ⋀ {Q}',          '{P} and {Q}', '{P} but {Q}'),
    Rule('{P} ⋁ {Q}',          '{P} or else {Q}', '{P} or {Q}'),
    ]

negations = [
    (word("not"), ""),
    (word("cannot"), "can"),
    (word("can't"), "can"),
    (word("won't"), "will"),
    (word("ain't"), "is"),
    ("n't", ""), # matches as part of a word: didn't, couldn't, etc.
    ]


def match_rules(sentence, rules, defs):
    """Match sentence against all the rules, accepting the first match; or else make it an atom.
    Return two values: the Logic translation and a dict of {P: 'english'} definitions."""
    sentence = clean(sentence)
    for rule in rules:
        result = match_rule(sentence, rule, defs)
        if result: 
            return result
    return match_literal(sentence, negations, defs)
        
def match_rule(sentence, rule, defs):
    "Match rule, returning the logic translation and the dict of definitions if the match succeeds."
    output, patterns = rule
    for pat in patterns:
        match = re.match(pat, sentence, flags=re.I)
        if match:
            groups = match.groupdict()
            for P in sorted(groups): # Recursively apply rules to each of the matching groups
                groups[P] = match_rules(groups[P], rules, defs)[0]
            return '(' + output.format(**groups) + ')', defs
        
def match_literal(sentence, negations, defs):
    "No rule matched; sentence is an atom. Add new proposition to defs. Handle negation."
    polarity = ''
    for (neg, pos) in negations:
        (sentence, n) = re.subn(neg, pos, sentence, flags=re.I)
        polarity += n * '～'
    sentence = clean(sentence)
    P = proposition_name(sentence, defs)
    defs[P] = sentence
    return polarity + P, defs
    
def proposition_name(sentence, defs, names='PQRSTUVWXYZBCDEFGHJKLMN'):
    "Return the old name for this sentence, if used before, or a new, unused name."
    inverted = {defs[P]: P for P in defs}
    if sentence in inverted:
        return inverted[sentence]                      # Find previously-used name
    else:
        return next(P for P in names if P not in defs) # Use a new unused name
    
def clean(text): 
    "Remove redundant whitespace; handle curly apostrophe and trailing comma/period."
    return ' '.join(text.split()).replace("’", "'").rstrip('.').rstrip(',')


string = "If roses and carnations are not red, if violets are not blue, then petunias are purple"
x, y = match_rule(string,
           Rule('{P} ⇒ {Q}', 'if {P}, {Q}'),
           {})

print(f'output: {x}, def:{y}')

"""
def convert_sentence(s):
	print(s)
	sym_rule = match_rule(s, Rule('{P} ⇒ {Q}', 'if {P}, {Q}'),{})
	return sym_rule

print(convert_sentence("If roses are red then violets are purple"))



if __name__=="__main__":
	main()
"""