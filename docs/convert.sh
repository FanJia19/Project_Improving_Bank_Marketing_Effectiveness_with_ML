#!/bin/bash

cd docs/content
# The templates have issues with references (unwanted 00)
# pandoc --citeproc *.md --pdf-engine=xelatex --metadata-file=../format/manual.yaml --template=../template/manual.latex -o ../bank_marketing.pdf
pandoc *.md --citeproc --pdf-engine=xelatex --metadata-file=../format/manual.yaml --template=../template/default.latex -o ../bank_marketing.pdf
# pandoc --citeproc *.md --pdf-engine=xelatex --metadata-file=../format/manual.yaml -o ../bank_marketing.pdf
