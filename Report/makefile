CONTENT = $(wildcard content/*.tex)
APPENDIX = $(wildcard appendix/*.tex)
IMG = $(wildcard img/**/*.*)
DOC = $(wildcard doc/**/*.*)

main.pdf: main.tex $(CONTENT) $(APPENDIX) $(IMG) $(DOC) biblo.bib
	@echo "" | pdflatex main
	@echo "" | biber main
	@echo "" | pdflatex -interaction=batchmode main
	@echo "" | pdflatex -interaction=batchmode main
	make wordCount
	
clean:
	-rm -r *.aux *.bcf *.log *.out *.toc *.xml *.bbl *.blg
	-rm -r ./*/*.aux
	-rm main.pdf

wordCount:
	@echo "Word Count: \c"
	@texcount -inc -1 -sum main.tex