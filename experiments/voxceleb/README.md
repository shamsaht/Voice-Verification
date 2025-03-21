In this module, we perform all experiments on the voxceleb test set. Reported under the section 8.8 : Optional: Analysis of Benchmark Reference Embeddings. 

dataset_stats.py : 
in this file, we compute general statistics about how many speakers and how many utternaces each speaker gets. The output is a latex table that includes the statistics.

test_scores.py : 
calculates popular voice verification metrics for voxceleb dataset to be reported as a baseline. 

clustering_score.py : 
We calculates the Silhouette score for the clusters in the test dataset, and generate a umap in 2D for the 40 speaker clusters