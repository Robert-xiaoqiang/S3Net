#!/bin/bash
if test $# -ne 1; then
	echo "iff 1 positional argument"
	exit 1
fi
for file in $1/img-[0-9]*.jpg; do
  # strip the postfix (".png") off the file name
  base_name=${file##*/}
  img_number=${base_name%%.jpg}
  number=${img_number#img-}
  # subtract 1 from the resulting number
  # i=$((number-1))
  # copy to a new name in a new folder
  mv ${file} $(printf "$1/00%s.jpg" $number)
  # mv ${file} $(printf "$1/%s.jpg" $number)
done
