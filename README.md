# Analysis Questions

1. What is the average length of a story in this collection? What is the shortest story (and how short it is)? What is the longest story (and how longis it)? Note that for this project, "short" and "long" are measured by the number of tokens, not the number of characters.

_Answer:_ The average length of a story in this collection is about 1215 tokens. The shortest story is 4 tokens long, and the longest story is 26139 tokens long

2. What word occurs in the most stories and how many stories does it occur in? What word has the largest number of occurrences and how many does it have?

_Answer:_ The word "the" appears in the most stories in the collection, with an occurrence of 966 documents. It also has the largest number of occurrences, with a total of 473568 in all the documents.

3. How many unique words are there in this collection? How many of them occur only once? What percent is that? Is that what you would expect? Why or why not?

_Answer:_ The number of unique words in this collection is 27217. There are 12497 words that occurs only once, which has a percentage of about 1.054% in the collection. This is what I would expect because given how many documents and how many number of words in the documents, there can only be a small percentage of them that is unique.

4. Run the query amherst college -- where that means the two words separately and not the phrase -- using either BM25 or QL (your choice). For any 10 of the top 50 top ranked documents, look at the text of the document and mark whether it is relevant. Put your judgments in a file called amherst-YOURUSERNAME.qrels Your should include the 10 storyIDs and a judgement of relevant that is 0 = has nothing to do with Amherst College, 1 = Amherst College is mentioned, or 2 = substantially relates to someone from or something that happened at Amherst College. Use the qrels file format from P2, with the queryname being "amherst", then the skip value of 0, then the storyID (NOT your internal docid), and then the 0/1/2 judgment, one of those per line.

_Answer:_ The file `amherst-anhl.qrels` contains the answer to this question.
