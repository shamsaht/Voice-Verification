In this directory we perform abilation studies
First abilitation study is to select all combinations of 3
files per user and check which one is the best for the purpose of generating 
reference utterances

Files for this study : 
1 - combination_analysis.py
2 - pickle_csv.py
3 - analysis_embed.py
4 - retrieve_combinations.py

combination_analysis.py : 
Optimized for multi-core cpu. TODO:GPU support
This file considers all combinations of 3 utterances as 
candidates for reference embeddings. Evaluates them against
all other utterance files, stores the following information 
per utterance. 
1 - reference_embeddings
2 - std of (1)
3 - for each speaker [mean,std,min,max] of its similarity with this utterance
4 - combination ID
All of those are stored in pickle files. this code actually generates multiple files
all for the purpose of analysis. But it represent the full experiment. All files generated 
by this script are, for N speakers
1 - N i_combination.pkl : combination_id : [files]
2 - N i_stats.pkl : whats described above
3 - speakermap.txt : speaker_id : speaker
1, and 2 since they contain one file per user are naturally stored in directories
All outputs are stored in the logs directory, directory outputs are child directories of 
the logs directory with sensible naming you'll find them easily. 

pickle_csv.py : 
This file takes all the pickle files inside the logs/stats and extract their 
info into csv files containing the same data. Easier for data analysis tho 
not necessary and could be skipped. Stores outputs in logs/csv

analysis_embed.py :
This one analyse the performance of each of the embeddings based on the 
distance between the mean of the speaker similarity scores (cluster), 
and the nearest speaker's mean (cluster). This is done based on :
distance = [speaker_mean - 2 * speaker_std] - [others_mean + 2 * others_std]
Obviously, the best reference utterance is one that has the max distance, while 
a poor representation is one with the lowest score. In this file we report both
the best and the worst performing combinations based on this score. Other studies might consider different distance equations. 

retrieve_combinations.py : 
This file, given a combination number and a speaker ID. retrieves the files making up this combination. The geneatred embedding, and generate a pkl file with one reference utterance per speaker. When the verification system is used, it should take the reference utterances from this file. Also, the examples producing these utterances are removed from the dataset stored in srta_vauth. If no speaker, nor a combination is provided, this will automate all the processes above to analayse, pick, and generate the reference embeddings. 

best_embed.py :
this function includes the updated retrieval function for the embeddings.

To run experiment 1. Best Embeddings. You can run the file retrieve_combinations.py with no parameters passed. It will do everything from the beggining. But if you want to skip the first couple of steps you can edit the phase parameter in the file. 