#1/usr/bin/env python
import music
import tunings

print "Lets find a tuning!"
t = tunings.way_low
print t
print "This is actually really fun, but would require re-stringing the guitar (trust me, I've done it, you need thick strings, and thicker skin)."
print "So lets transpose it 6-semi-tones up to D2, which is totally reasonable..."
t2 = [x.transpose(6) for x in t]
print t2
print "I wonder what intervals are in this bad-boy?"
intervals = [music.Interval(t2[0], x) for x in t2]
print intervals
print "...or some cool root-names (without octave information)"
print [x.name for x in intervals]
print "at some point I'll probably add a chord table, but until then, cya!"
