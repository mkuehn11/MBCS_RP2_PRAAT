# MBCS_RP2_PRAAT

Analysis code for Research Project 2 - PRAAT study at the UMC Utrecht 2020-2021. The corresponding clinical data is not publicly available, which is why file paths and any identifyable information has been removed. 

## Main Analysis: Group Comparison -- full_analysis_03_21.ipynb

1. Start by supplying the paths to the installations and files

`
            os_sep = os.path.abspath(os.sep)
            opensmile_dir = os.path.join(os_sep, '') #openSMILE installation location
            praat_path = os.path.join(os_sep, 'Applications','Praat.app', 'Contents', 'MacOS', 'Praat') # PRAAT execetuable
            audio_dir = os.path.join(os_sep, '') # audio data
            data_dir = os.path.join(os_sep, '') #csv output
            dem_dir = os.path.join(os_sep, '') #txt files that contain the subject id's of the participants in each group
`

The **txt files with the participant ids and extracted csv files are available upon request**, if the csv files are already created you can skip to step 3, detailed explanations of each step are supplied in the Jupyter Notebook containing the analysis code.

1.1 Run the following cells to generate the directories and split the audio files into the separate audio channels for each speaker.

`
            #define and create directories
            split_ch_output = os.path.join(audio_dir, 'split_channels')
            temp_dir = os.path.join(audio_dir, 'tmp')

`

Skip the following if you already have separate files with the structure of ´audio_name_ch1.wav audio_name_ch2.wav´ and move these files into the `os.path.join(audio_dir, 'split_channels')` directory which was created.

`
            #run praat script on batch with arguments
            subprocess.call([praat_path, 
                        '--run',
                        split_script,     #path to script
                        temp_dir + os_sep, #input dir + praat needs the slash at the end of a path
                        split_ch_output + os_sep]) #output dir
`
            
1.2 Run the PRAAT script to annotate the speaking intervals in each file and create concatenated audio files containing only the speaker intervals.

`    
            #run praat script on batch
            subprocess.call([praat_path, 
                            '--run',
                            concat_script,    #path to script
                            temp_dir + os_sep, #praat needs the slash at the end of a path
                            concat_ch_output + os_sep, #output audio
                            turn_textgrids + os_sep])  #output textgrids
`

Similarly, if the TextGrid files already exist, copy them into `os.path.join(data_dir, 'textgrids', 'turn_textgrids')`


2. Feature extraction with openSMILE

2.1 Run the cells to extract the features for each of the audio files with openSMILE if the installation was correct and the audio files in `split_channels` and TextGrids in `turn_textgrids` are provided.

`
            #run openSMILE extraction with arguments
            subprocess.run(['SMILExtract', 
                            '-C', config_file, 
                            '-I', file, 
                            '-csvoutput', output_file,
                            '-start', str(start),
                            '-end', str(stop),
                            '-instname', instname])
`

If the feature CSV files are already created, make sure they exist in `os.path.join(data_dir, 'opensmile', 'egemaps_summary_turns')`.
      

3. Analysis

3.1 Annotation cleanup. To exclude some falsely labeled intervals, speaking intervals with zero pitch are excluded from the csv files and saved in `os.path.join(data_dir, 'opensmile', 'egemaps_summary_turns_zero_filtered')`. 

3.2 Synchrony measures, for each of the audio features (pitch, loudness etc) the synchrony between speakers is calculated as `spearman r`

`
            for feature, rows in feature_rows.items():

                df = pd.DataFrame(rows)

                df.to_csv(os.path.join(summary_dir, feature + '_summary.csv'), sep = ';')
`
3.3 Compare groups based on the groupings in the supplied txt files (i.e. Contols vs. Patients, Male vs. Female). Dataframes with the p and t values are saved in `os.path.join(dem_dir, 'ttest_groups.csv')` for between groups and `dem_dir, 'one_sided_groups.csv')` for within groups.

`
            idxs_g1 = getIndices(df, group1) #the matching subjects in the dataframe
                        idxs_g2 = getIndices(df, group2)

                        values_g1 = df[df['soundname'].isin(idxs_g1)]['r_z']   #select converted r value
                        values_g2 = df[df['soundname'].isin(idxs_g2)]['r_z']          

                        t, p = stats.ttest_ind(values_g1, values_g2, equal_var = False) #equal var = False --> Welch's t-test

                        row['T'] = t
                        row['p'] = p
`

## Secondary Analysis --> comparisons_over_time.ipynb

1. After running the main analysis script these should be available or supplied:

`
            dfs = os.path.join(os_sep, wd, 'opensmile', 'egemaps_summary_turns_zero_filtered') #the feature dfs of the interviews
            dem_dir = os.path.join(os_sep, '') #where to find the txt files with the group information of each participant

`

1.1. Split the csv with features for each turn into two halves:

`
            first_half, second_half = np.array_split(df.index, 2)

                if 'ch1' in file:
                    ch1_first_half.append(df.loc[first_half])
                    ch1_second_half.append(df.loc[second_half])

                else:
                    ch2_first_half.append(df.loc[first_half])
                    ch2_second_half.append(df.loc[second_half])
`
1.2 Calculate the synchrony as `spearman r` again as in the first part but now for each conversation half separately:

`
            feature_rows_first_half = calculateSynchronyFromDF(ch1_first_half, ch2_first_half, features)
            feature_rows_second_half = calculateSynchronyFromDF(ch1_second_half, ch2_second_half, features)

`

1.3. Load the supplied group information 
`

            controls = np.loadtxt(os.path.join(dem_dir, 'control_subs.txt'), dtype= str)
            patients = np.loadtxt(os.path.join(dem_dir, 'patient_subs.txt'), dtype= str)

`

1.4. Compare the first and second half with a paired t-test for each group:

`
                    cond1 = dfs_condition1[feature]
                    cond2 = dfs_condition2[feature]

                    idxs_group = getGroupIndices(cond1, group) #the matching group subjects in the dataframe

                    x = cond1[cond1['soundname'].isin(idxs_group)]['r_z']   #select converted r value
                    y = cond2[cond2['soundname'].isin(idxs_group)]['r_z']  

                    #paired ttest!
                    t, p = stats.ttest_rel(x, y)

                    row['T'] = t
                    row['p'] = p
`

The uncorrected (!) results are displayed in the Notebook.
        
1.5. The extact same process is repeated, but now interviews are split into three parts

`

            first, second, third  = np.array_split(df.index, 3)
`

The uncorrected p and t values are printed and plotted in the Notebook.


## Exploratory Analysis -- log_regression.ipynb

1.1 The group information, feature dataframes and synchrony measures should be available after running the main analysis or upon request.

`
            controls = np.loadtxt('control_subs.txt', dtype= str)
            dfs = os.path.join(os_sep, wd, 'opensmile', 'egemaps_summary_turns_zero_filtered') #feature dataframes
            #load synchrony dataframes
            pitch = pd.read_csv(os.path.join(summary_dir, 'F0semitoneFrom27.5Hz_sma3nz_amean_summary.csv'), sep = ';', index_col = [0])

`

1.2 As predictors for the regression model, all sychrony coefficients are added to the feature dataframe used as indepednent variables.

`
            feature_df = pd.DataFrame([pitch['r'], loudness['r'], syll['r'], pause['r'], pitch_var['r']]).T
            feature_df.columns = ['pitch', 'loudness', 'syll_rate', 'pause_dur', 'pitch_var']
            feature_df.index = pitch['soundname']

`

1.2 Additionally as sanity checks or for exploration, the years of education and the interviewer are loaded, numerically encoded, and can be added to the predictors.

`
            feature_df['interviewer'] = interviewers_encod
            feature_df['yoe'] = yoe

`

1.3 The target variable is the group status of each observation (each interview)

`

            group_df = pd.DataFrame(group_encoding, columns = ['group'])

`

1.4 Independence between predictors can be explored using Variance Inflation Factor

`

            variance_inflation_factor(X_intcpt.values)
`

1.5. Split the data into stratified train and test sets

`
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0, stratify = y)
`

1.6. The ranking of the importance of each feature can be explored using Recursive Feature Elimination if that information is viable or if predictors need to be dropped.

`

            rfe_mod =  RFECV(estimator = LogisticRegression(), step=1, scoring = 'roc_auc')

            rfe_fit = rfe_mod.fit(X_train, np.array(y_train).flatten())

`

1.7. Train the model for k folds, i.e. 5

`
            cv = StratifiedKFold(n_splits = 5, shuffle = True)

            for train, test in cv.split(X_arr, y_arr):

                log_fit = log_sklearn.fit(X_arr[train], y_arr[train])

                yhat_proba = log_fit.predict_proba(X_arr[test])

`

The results of the performance are plotted in the Notebook.

1.7. Train the full model without k-folding to evaluate full performance and plot the confusion matrix

`
            log_fit = log_sklearn.fit(X_train, np.array(y_train).flatten())

`

1.8. Inspect the false positives and compare their negative symptom scores.

`
            #identify which patients were correctly or incorrectly classified
            incorrect_ident = real_vs_prediction.loc[(real_vs_prediction['group'] == 1) & (real_vs_prediction['yhat'] == 0)]
            correct_ident = real_vs_prediction.loc[(real_vs_prediction['group'] == 1) & (real_vs_prediction['yhat'] == 1)]

`

Results are printed and plotted in the Notebook.
