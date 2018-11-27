#!/usr/bin/env bash

if [[ -z $3 ]]; then
  echo "Usage: sari output reference src"
  exit 1
fi

set -u

format=plain
grep "|||" $1 > /dev/null
if [[ $? -eq 0 ]]; then
  format=nbest
fi

output=$1
refs=$2
src=$3
class=$4
numrefs=1
#output=$1
#refs=$2
#src=$3
#numrefs=1

java -cp $class -Dfile.encoding=utf8 -Djava.library.path=lib -Xmx256m -Xms256m -Djava.util.logging.config.file=logging.properties joshua.util.JoshuaEval -cand $output -format $format -ref $refs -rps $numrefs -m STAR 4 $src -v 0
