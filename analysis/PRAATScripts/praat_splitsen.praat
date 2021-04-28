#Alban Voppel, UMC
#select list of files in folder
#this script runs on all the audio files in the folder and saves them in the specified folder with a new name.

form Specify Directories
	sentence inputdir /audio_path/
	sentence outputdir /output_path/
	sentence textgriddir /output_for_textrgrids/
endform

list = Create Strings as file list: "Filelist", inputdir$ + "*.wav"
n = Get number of strings

for i to n
	#create all the variables from the filename in the list.
	selectObject: list
	filename$ = Get string: i
	basename$ = filename$ - ".wav"
	textgrid1$ = "TextGrid " + basename$ + "_ch1"

	textgrid1$ = replace$ (textgrid1$, ".", "_", 3)
	ch1name$ = "Sound " + basename$ + "_ch1" 
	ch2name$ = "Sound " + basename$ + "_ch2"
	ch1name$ = replace$ (ch1name$, ".", "_", 3)
	ch2name$ = replace$ (ch2name$, ".", "_", 3)

	#load in the file as a sound, extract 2 channels.
	sound = Read from file: inputdir$ + filename$
	Extract all channels
	
	selectObject: ch1name$
	To TextGrid (silences): 100, 0, -25, 1, 0.1, "interviewer_silent", "interviewer_speaks"
	selectObject: textgrid1$, ch2name$ 
	Extract intervals where: 1, "no", "is equal to", "interviewer_silent"
	
	#count and save all the sound intervals for later deletion.

	total_parts = numberOfSelected()
	for x to total_parts
  		part[x] = selected(x)
	endfor
	
	Concatenate recoverably
	selectObject: "Sound chain"
	nowarn Save as WAV file: outputdir$ + filename$ + "_ch2_interv_removed.wav"
	
	#remove all the sound segments from earlier
	for x to total_parts
  		plusObject: part[x]
	endfor
	Remove

	#######
	#do the same for channel 1
	selectObject: textgrid1$, ch1name$

	Extract intervals where: 1, "no", "is equal to", "interviewer_speaks"

	total_parts = numberOfSelected()
	for x to total_parts
  		part[x] = selected(x)
	endfor

	Concatenate recoverably
	selectObject: "Sound chain"
	nowarn Save as WAV file: outputdir$ + filename$ + "_ch1_pat_removed.wav"

	#remove all the sound segments from earlier
	for x to total_parts
  		plusObject: part[x]
	endfor
	Remove

	selectObject: textgrid1$
	Save as text file: textgriddir$ + filename$ + ".TextGrid"

	#######

	#clean up some other variables left over.
	removeObject: "TextGrid chain", sound, ch1name$, ch2name$, textgrid1$
endfor
