from src.query import Query

queries = {
    'rsa' : Query('rsa', ['x'], [('R', ['x', 'y']), ('S', ['x', 'z']), ('A', ['x'])]),
    'hc-star-3': Query('long-hc', ['x'], [('A', ['x']),('R', ['x', 'a']), ('S', ['x', 'b']), ('T', ['x', 'c'])]),
    'hc-star-5': Query('long-hc', ['x'], [('A', ['x']),('R', ['x', 'a']), ('S', ['x', 'b']), ('T', ['x', 'c']), ('U', ['x', 'd']), ('V', ['x', 'e'])]),
    'rays-5': Query('rays-5', ['x'], [('R', ['x', 'a']), ('S', ['x', 'b']), ('T', ['x', 'c']), ('U', ['x', 'd']), ('V', ['x', 'e'])]),
    'rays-3': Query('rays-3', ['x'], [('R', ['x', 'a']), ('S', ['x', 'b']), ('T', ['x', 'c'])]),
    'hc-sj-ucq': Query('hc-sj-ucq', ['x'], [('R', ['x', 'a','b']), ('R', ['x', 'b','c']), ('R', ['x', 'c', 'a']), ('R', ['x', 'e', 'f']), ('R', ['x', 'f', 'g'])]),
}