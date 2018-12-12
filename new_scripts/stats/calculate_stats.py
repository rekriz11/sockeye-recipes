# You will need to `pip3 install textstat` and `git clone https://github.com/jhclark/tercom.git` (be sure to build the jar)
import argparse
import sys
import os
import subprocess
import re
import textstat

parser = argparse.ArgumentParser(description='Calculate descriptive stats.')
parser.add_argument('-c', '--complex', required=True, help='path to complex reference')
parser.add_argument('-s', '--simple', required=True, help='path to simple or a file with a list of files to evaluate against')
parser.add_argument('-f', '--isfile', action='store_true', help='use this flag if you are passing in a file with a list of files to -s')
parser.add_argument('-d', '--directory', default=None, help='path prefix to the simple files if -f is used')
parser.add_argument('-o', '--out', default=None, help='path to output file, if not given print to stdout')

args = vars(parser.parse_args())

complex_temp = 'complex.temp'
simple_temp = 'simple.temp'

outfile = None
if args['out'] is not None:
    outfile = open(args['out'], 'w')

with open(args['complex'], 'r') as complex_file:
    complex_sents = complex_file.readlines()

n = len(complex_sents)

if args['isfile']:
    with open(args['simple'], 'r') as file:
        simple_fns = [x.strip() for x in file.readlines()]
else:
    simple_fns = [args['simple']]

def write_out_str(str):
    print(str)
    if outfile is not None:
        outfile.write(f'{str}\n')

def write_out_stat(str, num):
    write_out_str(f'{str}: {round(num, 2)}')

def format_file(filename, lines):
    with open(filename, 'w') as file:
        for i, line in enumerate(lines):
            file.write(f'{line.strip()} ({i})\n')

def process_simple_file(simple_fn):
    if args['directory'] is not None:
        simple_fn = os.path.join(args['directory'], simple_fn)

    with open(simple_fn, 'r') as simple_file:
        simple_sents = simple_file.readlines()

    if len(simple_sents) != n:
        print(f'Complex had {n} lines but simple had {len(simple_sents)}', file=sys.stderr)
        return

    write_out_str(simple_fn)
    avg_length = sum([len(sent.split(' ')) for sent in simple_sents]) / n
    write_out_stat('Sentence Length', avg_length)

    word_length = sum([sum([len(w) for w in sent.split(' ')])/len(sent.split(' ')) for sent in simple_sents]) / n
    write_out_stat('Word Length', word_length)
    
    avg_fkscore = sum([textstat.flesch_kincaid_grade(sent) for sent in simple_sents]) / n
    write_out_stat('Flesch-Kincaid', avg_fkscore)

    # TER
    format_file(simple_temp, simple_sents)
    command_string = f'java -jar tercom/tercom-0.10.0.jar -r {complex_temp} -h {simple_temp} -o sum -n out'
    #custom = '-N'
    #command_string += ' ' + custom
    subprocess.run(command_string.split(' '), stdout=subprocess.DEVNULL)

    with open('out.sum', 'r') as file:
        nums = [float(x) for x in re.findall('\d+.?\d+', file.readlines()[-1])]

    write_out_stat('Inserts', nums[0] / n)
    write_out_stat('Deletes', nums[1] / n)
    write_out_stat('Subs', nums[2] / n)
    write_out_stat('Shifts', nums[3] / n)
    write_out_stat('TER', nums[-1] / 100)

format_file(complex_temp, complex_sents)

for simple_fn in simple_fns:
    process_simple_file(simple_fn)

# Cleanup
os.remove(simple_temp)
os.remove(complex_temp)
if outfile is not None:
    outfile.close()
