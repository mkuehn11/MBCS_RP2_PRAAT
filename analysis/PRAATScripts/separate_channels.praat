#### split wav files in directory in separate channels ####

form Specify paths
	sentence inputdir /audio_directory/
	sentence outputdir /output_directory/
endform

list = Create Strings as file list: "Filelist", inputdir$ + "*.wav"
n = Get number of strings

for i to n
	#create all the variables from the filename in the list.
	selectObject: list
	filename$ = Get string: i
	basename$ = filename$ - ".wav"

	ch1name$ = "Sound " + basename$ + "_ch1" 
	ch2name$ = "Sound " + basename$ + "_ch2"
	ch1name$ = replace$ (ch1name$, ".", "_", 3)
	ch2name$ = replace$ (ch2name$, ".", "_", 3)

	#load in the file as a sound, extract 2 channels.
	sound = Read from file: inputdir$ + filename$
	Extract all channels
	
	selectObject: ch1name$
	nowarn Save as WAV file: outputdir$ + filename$ + "_ch1.wav"

	selectObject: ch2name$
	nowarn Save as WAV file: outputdir$ + filename$ + "_ch2.wav"

	removeObject: sound

	endfor
	