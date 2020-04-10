# beam_search
General use code for beam_search and EFN search

Basically, beam_search.py is the focus on this project. It contains 2 functions that can be used for any Language Model to perform a beam search and an EFN search. They can be called with beam_search.beam_search(logit_function, context) and beam_search.efn_search(logit_function, context), where logit_function is a function that accepts in a list of contexts, and outputs a list of logits, and context is the list of tokens to be the base of the search.

EFN (Expanding Frontier Nodes) Search is similar to beam_search, but instead of taking the most probable list of tokens, and discarding the rest, there is a stack of context lists, and we continually choose the most likely list of contexts, and finds the top_k context lists based off of this most-likely list. Basically, it is a breadth based version of beam search, rather than being depth based.
