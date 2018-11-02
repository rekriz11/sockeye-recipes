# Prints translated sentences with named entities not present in the source.
import os
import re
import sys

if len(sys.argv) != 3:
    print('Usage: check_named_entities.py [path to output] [path to input]', file=sys.stderr)
    sys.exit(1)

NAMED_ENTITY_PATTERN = re.compile(r'(PERSON|ORGANIZATION|LOCATION)@(\d)')

output_file = open(sys.argv[1], 'r')
input_file = open(sys.argv[2], 'r')
#new_input_file = open('./new_in.txt', 'w')
line_count = 1

while True:
    output_line = output_file.readline()
    input_line = input_file.readline()

    output_named_entities = frozenset(re.findall(NAMED_ENTITY_PATTERN, output_line))
    input_named_entities = frozenset(re.findall(NAMED_ENTITY_PATTERN, input_line))

    only_output_entities = [item for item in output_named_entities if item not in input_named_entities]

    if len(only_output_entities) > 0:
        print(f'[{line_count}] {only_output_entities}')
        print(f'Output: {output_line}Input: {input_line}')
        #new_input_file.write(input_line)

    if output_line == '' or input_line == '':
        break

output_file.close()
input_file.close()
#new_input_file.close()
