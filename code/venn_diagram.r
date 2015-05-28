# From: http://manuals.bioinformatics.ucr.edu/home/R_BioCondManual#TOC-Venn-Diagrams

#################
## Sample data ##
#################
source("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/My_R_Scripts/overLapper.R") # Imports required functions.
setlist <- list(A=sample(letters, 18), B=sample(letters, 16), C=sample(letters, 20), D=sample(letters, 22), E=sample(letters, 18), F=sample(letters, 22, replace=T))
   # To work with the overLapper function, the sample sets (here six) need to be stored in a list object where the different
   # compontents are named by unique identifiers, here 'A to F'. These names are used as sample labels in all subsequent data
   # sets and plots.
sets <- read.delim("http://faculty.ucr.edu/~tgirke/Documents/R_BioCond/Samples/sets.txt")
setlistImp <- lapply(colnames(sets), function(x) as.character(sets[sets[,x]!="", x]))
names(setlistImp) <- colnames(sets)
   # Example how a list of test sets can be imported from an external table file stored in tab delimited format. Such
   # a file can be easily created from a spreadsheet program, such as Excel. As a reminder, copy & paste from external
   # programs into R is also possible (see read.delim function).
OLlist <- overLapper(setlist=setlist, sep="_", type="vennsets"); OLlist; names(OLlist)


#########################
## 4-way Venn diagrams ##
#########################
setlist4 <- setlist[1:4]
OLlist4 <- overLapper(setlist=setlist4, sep="_", type="vennsets")
# from running: `python code/4way_test_ciresan2012.py 0 ciresan2012_bs12_nw14_d1_4Layers_cc1.pkl ciresan2012_bs12_nw16_d1_4Layers_cc1.pkl ciresan2012_bs12_nw18_d1_4Layers_cc1.pkl ciresan2012_bs12_nw20_d1_4Layers_cc1.pkl`
mines = c(0.004807692307692308, 0.0038060897435897435, 0.004006410256410256, 0.0033052884615384615, 0.0038060897435897435, 0.003205128205128205, 0.0030048076923076925, 0.0033052884615384615, 0.003205128205128205, 0.002904647435897436, 0.0030048076923076925, 0.002904647435897436, 0.002904647435897436, 0.0030048076923076925, 0.002904647435897436)
vennPlot(counts=round(mines * 10000), mysub="Left to Right Normalized width = {14,16,18,20}", yoffset=c(0.3, -0.2), mymain="Committee error in Basis Points")
