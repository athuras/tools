# Hacked together a class to make transpositions easier.
from collections import OrderedDict


_master_scale = ['C', 'C#', 'D', 'Eb',
                 'E', 'F', 'F#', 'G',
                 'G#', 'A', 'Bb', 'B']

_interval_names = OrderedDict([
    ('Unison', 'P1'),
    ('Minor second', 'm2'),
    ('Major second', 'M2'),
    ('Minor third', 'm3'),
    ('Major third', 'M3'),
    ('Perfect fourth', 'P4'),
    ('Augmented fourth', 'A4'),
    ('Perfect fifth', 'P5'),
    ('Minor sixth', 'm6'),
    ('Major sixth', 'M6'),
    ('Minor seventh', 'm7'),
    ('Major seventh', 'M7')])

mirror_table = {v: k for k, v in _interval_names.iteritems()}
_keys = [k for k in _interval_names]

class Interval(object):
    '''Generated from two Note objects'''
    _interval_table = {  # Interval info, Inverse Interval info
        i + 1: (_interval_names[k], _interval_names[_keys[-(i + 1)]])
        for i, k in enumerate(_keys[1:])}
    _interval_table.update({0: ('P1', 'P1')})

    def __init__(self, n1, n2):
        self.delta = (n2.idx + n2.octave * len(_master_scale)
                    - n1.idx - n1.octave * len(_master_scale))
        q, r = divmod(self.delta, len(_master_scale))
        self.inverse_flag = True if self.delta < 0 else False
        self.normed = abs(r)
        self.octaves = q

    @property
    def abv(self):
        return Interval._interval_table[self.normed][self.inverse_flag]

    @property
    def inv_abv(self):
        return Interval._interval_table[self.normed][not self.inverse_flag]

    @property
    def name(self):
        return mirror_table[self.abv]

    @property
    def inv_name(self):
        return mirror_table[self.inv_abv]

    def __repr__(self):
        return 'Interval: M%s::%s' % (self.octaves, self.abv)


class Note(object):
    def __init__(self, s):
        '''Input in the format <letter>(<accidentals>)<integer>'''
        octave = int(s[-1])
        letter = s[0].upper()
        accidentals = s[1:-1]
        aug = 0
        if len(accidentals) == 1:
            letter += accidentals[0]
        else:
            for a in accidentals:
                types = {'b': -1, '#': 1}
                aug += types[a]

        if aug > 0 and letter + '#' in _master_scale:
            aug -= 1
            letter += '#'
        elif aug < 0 and letter + 'b' in _master_scale:
            aug += 1
            letter += 'b'

        if letter not in _master_scale:
            aug += {'b': -1, '#': 1}[letter[1]]
            letter = letter[0]

        self.idx = _master_scale.index(letter)
        self.octave = octave
        self.increment(aug)

    def increment(self, i=1):
        '''Transpose note by i-semitones'''
        q, r = divmod(self.idx + i, len(_master_scale))
        self.octave += q
        self.idx = r

    def decrement(self, i=1):
        return self.increment(-i)

    def transpose(self, i):
        '''Returns a new Note object transposed by i-semitones'''
        q, r = divmod(self.idx + i, len(_master_scale))
        return Note.new_note(r, q + self.octave)

    @classmethod
    def new_note(cls, idx, octave):
        x = Note('A1')
        x.idx = idx
        x.octave = octave
        return x

    @property
    def pitch(self):
        return _master_scale[self.idx] + str(self.octave)

    def __repr__(self):
        return 'Note: ' + self.pitch
