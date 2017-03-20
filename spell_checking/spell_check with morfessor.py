import Levenshtein
import morfessor
io = morfessor.MorfessorIO()

model_types = io.read_binary_model_file('C:/Users/Ольга/PycharmProjects/DSM_morphology/morfessor/types')

word_segments = [word_and_freq[0] for word_and_freq in model_types.get_constructions()]
"""for elem in model_types.get_segmentations():
    print(elem)"""
word = 'схездил'
corr_word = 'съездил'
print(model_types.viterbi_nbest(corr_word,3))
print(model_types.viterbi_nbest(word, 3))

segmentation_trial = model_types.viterbi_nbest("схездил", 1)[0][0]
candidates = []
candidates_ed = []
output = []
closest_candidates = []
for segment in segmentation_trial:

    for word_segment in word_segments:
        len_segment = len(word_segment)
        edit_distance = Levenshtein.distance(segment, word_segment)/len(word_segment)
        candidates.append(word_segment)
        candidates_ed.append(edit_distance)
    minimum = min(candidates_ed)
    output = [candidates[i] for i in range(len(candidates)) if candidates_ed[i] == minimum]
    #print(output)

    jaro_distance_list = []
    for word_segment in output:
        jaro_distance = 1 - Levenshtein.jaro_winkler(segment, word_segment)
        jaro_distance_list.append(jaro_distance)

    minimum_jaro = min(jaro_distance_list)
    closest_candidates.append([output[i] for i in range(len(output)) if jaro_distance_list[i] == minimum_jaro])

print(closest_candidates)