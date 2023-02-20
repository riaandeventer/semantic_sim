# Software Engineer: RIAAN DEVENTER
# Written on 15 February 2023

#------------------------------------------------------------------------------------------------
# Run code extracts to showcase different models of Semantic Similarity (NLP)
# ● Write a note about what you found interesting about the similarities
#   between cat, monkey and banana and think of an example of your own.
#
# ● Run the example file with the simpler language model ‘en_core_web_sm’
#   and write a note on what you notice is different from the model 'en_core_web_md'.
#
# NOTE: The 'en_core_web_md' model is more advanced than the 'en_core_web_sm' model.
#       'en_core_web_md' have more data to compare to and the similarity percentage 
#       is much lower.
#------------------------------------------------------------------------------------------------

# ======= Working with spaCy ===== #

import spacy

nlp = spacy.load('en_core_web_md')

# Model - SIMILARITY WITH SPACY
# We are using the keyword ’similarity’ for getting the similarity between the words.
word1 = nlp("cat")
word2 = nlp("monkey")
word3 = nlp("banana")
print()
print(f"--------------- {word1.text}, {word2.text}, {word3.text} Similarities ---------------")
print(word1.text, word2.text + " - ", word1.similarity(word2))
print(word3.text, word2.text + " - ", word3.similarity(word2))
print(word3.text, word1.text + " - ", word3.similarity(word1))

'''
OUTPUT:
--------------- cat, monkey, banana Similarities ---------------
cat monkey -  0.5929930274321619
banana monkey -  0.40415016164997786
banana cat -  0.22358825939615987
'''
# NOTE: Similarities between cat, monkey and banana.
'''
cat, monkey has high similarity since they are both animals and mammals.
banana, monkey has high similarity since monkeys eat bananas.
banana,cat has lower similarity since cats don't eat bananas or monkey's food.
'''
# Another example.
word1 = nlp("cat")
word2 = nlp("dog")
word3 = nlp("fish")
print()
print(f"--------------- {word1.text}, {word2.text}, {word3.text} Similarities ---------------")
print(word1.text, word2.text + " - ", word1.similarity(word2))
print(word3.text, word2.text + " - ", word3.similarity(word2))
print(word3.text, word1.text + " - ", word3.similarity(word1))

'''
OUTPUT:
--------------- cat, dog, fish Similarities ---------------
cat dog -  0.8220816752553904
fish dog -  0.3187839199176165
fish cat -  0.31987174990274586
'''
# NOTE: Similarities between cat, dog and fish.
'''
--> This is an interesting one.
cat, dog has high similarity since they are both animals an mammals and both pets living in same environment.
fish, dog has lower similarity since dog and fish are both animals but not the same animal group.
fish, cat has more or less the same similarity as fish & dog which is strange. Even though the 2 relationships
    are pretty much the same, fish is a favorite food of cats that does not seem to be considered.
'''

# Model - WORKING WITH VECTORS
# In the case where you have a series of words and want to compare them all with
# one another, you can use the format outlined in this section.
print()
print(f"--------------- Word Series Similarities ---------------")
tokens = nlp('cat apple monkey banana ')
for token1 in tokens:
    for token2 in tokens:
        print(token1.text, token2.text + " - ", token1.similarity(token2))

'''
OUTPUT:
--------------- Word Series Similarities ---------------
cat cat -  1.0
cat apple -  0.2036806046962738
cat monkey -  0.5929930210113525
cat banana -  0.2235882580280304
apple cat -  0.2036806046962738
apple apple -  1.0
apple monkey -  0.2342509925365448
apple banana -  0.6646699905395508
monkey cat -  0.5929930210113525
monkey apple -  0.2342509925365448
monkey monkey -  1.0
monkey banana -  0.4041501581668854
banana cat -  0.2235882580280304
banana apple -  0.6646699905395508
banana monkey -  0.4041501581668854
banana banana -  1.0
'''

# Model - WORKING WITH SENTENCES
# We can work on similarity between longer sentences with below section.
print()
print(f"--------------- Longer Sentence Similarities ---------------")
sentence_to_compare = "Why is my cat on the car"
sentences = ["where did my dog go",
"Hello, there is my car",
"I\'ve lost my car in my car",
"I\'d like my boat back",
"I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)
for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)

print()
'''
OUTPUT:
--------------- Longer Sentence Similarities ---------------
where did my dog go -  0.630065230699739
Hello, there is my car -  0.8033180111627156
I've lost my car in my car -  0.6787541571030323
I'd like my boat back -  0.5624940517078084
I will name my dog Diana -  0.6491444739190607
'''
