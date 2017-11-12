from lxml import etree
from wordsegment import segment, load
from pyjarowinkler import distance
from nltk import word_tokenize, pos_tag
from random import randint

load()


def parse_xml(xml_file):
    subjs, rels, objs = [], [], []
    comms = []

    with open(xml_file, encoding='utf-8') as fobj:
        xml = fobj.read()

    root = etree.fromstring(xml)

    entries = root.getchildren()[0]  # cos there's only one entries elem

    for entry in entries:
        children = entry.getchildren()

        comms.append([])
        for child in children:

            if child.tag != 'originaltripleset':

                if child.tag == 'modifiedtripleset':

                    for subchild in child.getchildren():
                        #triples.append([string.replace('_', ' ') for string in subchild.text.split(' | ')])
                        for j, string in enumerate(subchild.text.split(' | ')):
                            str_to_save = string.replace('_', ' ')
                            if j == 0:
                                subjs.append(str_to_save.replace('.', ''))
                            elif j == 1:
                                rels.append(' '.join(segment(str_to_save)))
                            elif j == 2:
                                objs.append(str_to_save)

                elif child.tag == 'lex':

                    if child.get('comment') == 'good':
                        comms[-1].append(child.text)

    return subjs, rels, objs, comms


subjects, relations, objects, comments = parse_xml('train/1triples/1triple_allSolutions_SportsTeam_train_challenge.xml')

subj_set = set(subjects)
rels_set = set(relations)

"""Who is the manager of Lumezzane?"""

ask = True
while ask:
    question = input('please type in your question: ')
    text = word_tokenize(question)
    tags = pos_tag(text)

    subject_cands = []
    concat = False
    no_subj = False

    for w, t in tags:
        if t == 'NNP' or t == 'NNPS':
            if concat:
                subject_cands[-1].append(w)
            else:
                subject_cands.append([])
                concat = True
                subject_cands[-1].append(w)
        else:
            concat = False

    # looking for subjects
    proposed_subj = ''

    for ne in [' '.join(cand) for cand in subject_cands]:
        if ne in subj_set:
            proposed_subj = ne
            break
        else:
            max_dist = 0
            closest_subj = ''

            for subj in subj_set:
                dist = distance.get_jaro_distance(ne, subj, winkler=True, scaling=0.1)

                if dist > max_dist:
                    max_dist = dist
                    closest_subj = subj

            if max_dist > 0.75:
                proposed_subj = closest_subj
                break
            else:
                no_subj = True
                print('Sorry, I do not understand')
                print('maybe you were asking about', closest_subj, ', though I am not sure')
                break

    if not no_subj:
        # looking for relations
        potential_rels_id = []
        for i, subj in enumerate(subjects):
            if subj == proposed_subj:
                potential_rels_id.append(i)

        potential_rels = []
        for pot_rel_id in potential_rels_id:
            potential_rels.append(relations[pot_rel_id])

        success = False
        for rel in potential_rels:
            if rel in question:
                answers = comments[potential_rels_id[potential_rels.index(rel)]]
                index = randint(0, len(answers)-1)
                print('my answer:', answers[index])
                success = True
                break

        if not success:
            print('Sorry, I do not understand')
            print('you were asking about', proposed_subj, ', but I know little about it')

    response = input('Want to ask something else?')
    if response == 'No':
        ask = False
