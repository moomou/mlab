import re

COMMENT_RE = re.compile('\/\*[\s\S]*?\*\/|\/\/.*')

def replace_re_with(text, regex, replacement=' '):
    return re.sub(regex, replacement, text)

with open('./test.txt') as f:
    t = f.read()
    print(replace_re_with(t, COMMENT_RE, 'YES!'))
