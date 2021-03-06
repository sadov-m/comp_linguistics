# 08.10 Task 1: Affect in Tweets
## What was done
I've chosen the task EI-reg. The main goal was to detect emotion intensity (regression). There were 4 emotions (anger, fear, joy, or sadness). 

I decided to conduct two experiments for each of emotion category: the first one was to use TF-IDF vectorizer to transform all the data (preprocessing will be described in the next paragraph) and the second one - to select features manually (use words from hashtags, emojis and unordinary symbols and [SemEval-2016 English Twitter Mixed Polarity Lexicon](http://saifmohammad.com/WebPages/SCL.html), but the last part of features was not included in the final evaluation).

Preprocessing included deletion of users' nicknames, html-codes and stop-words.

As for the models, RandomForestRegressor, KNeighborsRegressor and Ridge were used.

All the results with graphics and models itself can be found here: report for [anger](https://github.com/sadov-m/comp_linguistics/tree/master/08.10%20Affect%20in%20Tweets/test_anger_results), [fear](https://github.com/sadov-m/comp_linguistics/tree/master/08.10%20Affect%20in%20Tweets/test_fear_results), [joy](https://github.com/sadov-m/comp_linguistics/tree/master/08.10%20Affect%20in%20Tweets/test_joy_results) and [sadness](https://github.com/sadov-m/comp_linguistics/tree/master/08.10%20Affect%20in%20Tweets/test_sadness_results). Cutting a long story short, transforming all the data allowed me to gain a small advantage over manually selected features, nevertheless it's quite marginal. Speaking about the models, Ridge Linear Model outperformed all the other models without even being tuned, which might prove the statement that RandomForestRegressor and KNeighborsRegressor don't perform well on sparse data. Full results for these tests are available only for anger emotion category: the others include only Ridge Linear Model performance track for two different approaches to feature selection mentioned in the beginning of this file.

As for the metrics, I didn't evaluate my systems using [evaluation script](https://github.com/felipebravom/EmoInt) because I thought that it would be strange to show Pearson correlation coefficient only (which is, by the way, might be too straight-forward due to the underlying assumption about the data being normally distribuited). So I decided to choose as metrics rho-value (Spearman correlation coefficient which is much more applicable regardless of data distribution), the coefficient of determination R^2 and RMSE.

The results can be replicated by running the models situated in the locations mentioned above and running the script called main_script.py setting "train" flag to False.