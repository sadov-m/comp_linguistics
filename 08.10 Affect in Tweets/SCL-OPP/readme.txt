Sentiment Composition Lexicon for Opposing Polarity Phrases (SCL-OPP)
Version 1.0
22 June 2016
Copyright (C) 2016 National Research Council Canada (NRC)
Contact: Saif Mohammad (saif.mohammad@nrc-cnrc.gc.ca)


***************************************
Terms of use
***************************************
1. This lexicon can be used freely for research purposes. 
2. The papers listed below provide details of the creation and use of the lexicon. If you use a lexicon, then please cite the associated papers.
3. If interested in commercial use of the lexicon, send email to the contact. 
4. If you use the lexicon in a product or application, then please credit the authors and NRC appropriately. Also, if you send us an email, we will be thrilled to know about how you have used the lexicon.
5. National Research Council Canada (NRC) disclaims any responsibility for the use of the lexicon and does not provide technical support. However, the contact listed above will be happy to respond to queries and clarifications.
6. Rather than redistributing the data, please direct interested parties to this page:
   http://www.saifmohammad.com/WebPages/SCL.html


Please feel free to send us an email:
- with feedback regarding the lexicon; 
- with information on how you have used the lexicon;
- if interested in having us analyze your data for sentiment, emotion, and other affectual information;
- if interested in a collaborative research project.


***************************************
General Description
***************************************

Sentiment Composition Lexicon for Opposing Polarity Phrases (SCL-OPP) is a list of single words and multi-word phrases and their associations with positive and negative sentiment. The phrases consist of two or three words. Each phrase includes at least one positive word and at least one negative word. The single words are taken from the set of words that are part of multi-word phrases. The words and phrases are drawn from tweets, and therefore include a small number of hashtag words and creatively spelled words.

The sentiment associations were obtained manually through crowdsourcing using the Best-Worst Scaling annotation technique.


***************************************
SemEval-2016
***************************************

Parts of this lexicon were used as development and test sets for SemEval-2016 shared task on Determining Sentiment Intensity of English and Arabic Phrases (Task 7) -- English Twitter Mixed Polarity Set (http://alt.qcri.org/semeval2016/task7/). All terms from SCL-OPP except the ones that were used in the previous competition (SemEval-2015 Task 10 Subtask E) and in other datasets of SemEval-2016 Task 7 were included. The SemEval-2016 set additionally included some same polarity phrases. In total, there were 1,269 terms. The sentiment association scores were converted into the range 0..1 for the SemEval competition. 



***************************************
File Format
***************************************

Each line in the file has the following format:

<term><tab><sentiment score><tab><POS pattern><tab><term freq>

where
<term> is a single word or a multi-word phrase; 
<sentiment score> is a real number between -1 and 1 indicating the degree of association of the term with positive sentiment;
<POS pattern> is a corresponding sequence of parts of speech (POS); 
<term freq> is the frequency of the term in the corpus of 11 million tweets.

The POS tags were determined by looking up the most common part-of-speech sequence for that term in a tweet corpus. The corpus was automatically POS tagged using the CMU Tweet NLP tool (http://www.cs.cmu.edu/~ark/TweetNLP/). For POS abbreviation, please see 
Kevin Gimpel, Nathan Schneider, Brendan O’Connor, Dipanjan Das, Daniel Mills, Jacob Eisenstein, Michael Heilman, Dani Yogatama, Jeffrey Flanigan, and Noah A. Smith. 2011. Part-of-speech tagging for Twitter: Annotation, features, and experiments. In Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL).


There are 1,178 terms: 602 single words and 576 phrases.


***************************************
More Information
***************************************

Details on the process of creating the lexicon can be found in:

Svetlana Kiritchenko and Saif M. Mohammad (2016) Happy Accident: A Sentiment Composition Lexicon for Opposing Polarities Phrases. Proceedings of the 10th edition of the Language Resources and Evaluation Conference (LREC), Portorož, Slovenia, 2016.

Svetlana Kiritchenko and Saif M. Mohammad (2016) Sentiment Composition of Words with Opposing Polarities. Proceedings of the 15th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL), San Diego, California, 2016.




