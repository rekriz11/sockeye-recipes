# You will need to `pip3 install textstat` and `git clone https://github.com/jhclark/tercom.git`
import sys
import os
import subprocess
import re
import textstat
from collections import Counter

if len(sys.argv) < 3:
    print('Usage: calculate_stats.py [complex] [simple] [optional: out]', file=sys.stderr)
    sys.exit(1)

with open(sys.argv[1], 'r') as complex_file:
    complex_sents = complex_file.readlines()
    
with open(sys.argv[2], 'r') as simple_file:
    simple_sents = simple_file.readlines()

if len(complex_sents) != len(simple_sents):
    print(f'Complex had {len(complex_sents)} lines but simple had {len(simple_sents)}', file=sys.stderr)
    sys.exit(1)

outfile = None
if len(sys.argv) > 3:
    outfile = open(sys.argv[3], 'w')

def write_out(str):
    print(str)
    if outfile is not None:
        outfile.write(f'{str}\n')

n = len(complex_sents)
avg_length = sum([len(sent.split(' ')) for sent in simple_sents]) / n
write_out(f'Length: {avg_length}')
avg_fkscore = sum([textstat.flesch_reading_ease(sent) for sent in simple_sents]) / n
write_out(f'Flesch-Kincaid: {avg_fkscore}')

# TER
complex_fn = 'complex.temp'
simple_fn = 'simple.temp'

def format_file(filename, lines):
    with open(filename, 'w') as file:
        for i, line in enumerate(lines):
            file.write(f'{line.strip()} ({i})\n')

format_file(simple_fn, simple_sents)
format_file(complex_fn, complex_sents)

command_string = f'java -jar tercom/tercom-0.10.0.jar -r {complex_fn} -h {simple_fn} -o sum -n out'
#custom = '-N'
#command_string += ' ' + custom
subprocess.run(command_string.split(' '), stdout=subprocess.DEVNULL)

with open('out.sum', 'r') as file:
    nums = [float(x) for x in re.findall('\d+.?\d+', file.readlines()[-1])]

write_out(f'Inserts: {nums[0] / n}')
write_out(f'Deletes: {nums[1] / n}')
write_out(f'Subs: {nums[2] / n}')
write_out(f'Shifts: {nums[3] / n}')
write_out(f'TER: {nums[-1] / 100}')

# Cleanup
os.remove(simple_fn)
os.remove(complex_fn)
if outfile is not None:
    outfile.close()
