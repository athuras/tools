# Some guitar tunings I like, formalized and transposable!
import music as M

def render(x):
    return [M.Note(y) for y in x]

standard = render(('E2', 'A2', 'D3', 'G3', 'B3', 'E4'))

# Something I've stumbled upon
way_low = render(('Ab1', 'Eb2', 'Ab2', 'B2', 'F#3', 'Bb3'))

# A Classic
dadgad = render(('D2', 'A2', 'D3', 'G3', 'A3', 'D4'))

# Don Ross' "Drac & Friends (part 1)"
dadgce = render(('D2', 'A2', 'D3', 'G3', 'C3', 'E3'))

# Don Ross' Berkeley Springs
facfcf = render(('F2', 'A2', 'C3', 'F3', 'C4', 'F4'))

# My Special Tuning
special = render(('C#2', 'Ab2', 'C#3', 'E3', 'B3', 'Eb4'))
