import numpy as np

def beam_search(get_logits, context, length = 30, beam_width = 3):
    """A beam search functions
    get_logits is a function that accepts context tokens and prints out logits for them
    context is an array of tokens (integers representing string fragments)
    length is the maximum number of tokens the context should be extended by
    beam_width is the top_k logits used in the beam search """
    
    max_length = len(context) + length
    frontier_contexts = [context]
    all_contexts = [context]
    probability_map = {}

    while(True):
        if (len(frontier_contexts[0]) == max_length):
            break
        
        for current_context in frontier_contexts:
            out_logits = get_logits(current_context)
            new_contexts = []

            # Normalize the outputs (Hopefully they have already been normalized, but just in case)
            out_logits = out_logits - np.max(out_logits)
            eo_logits = np.exp(out_logits) + 1e-20
            out_logits = np.log(eo_logits / (np.sum(eo_logits)))

            logit_indices = []
            logit_probs = []

            for _ in range(0, beam_width):
                logit_indices.append(np.argmax(out_logits))
                logit_probs.append(np.max(out_logits))
                np.put(out_logits, np.argmax(out_logits), np.min(out_logits))

            for i in range(len(logit_indices)):
                temp_context = current_context.copy()
                temp_context.append(logit_indices[i])
                if str(current_context) in probability_map:
                    probability_map[str(temp_context)] = probability_map[str(current_context)] + logit_probs[i]
                else:
                    probability_map[str(temp_context)] = logit_probs[i]

                new_contexts.append(temp_context)

            frontier_contexts = new_contexts
            new_probs = {}
            for con in frontier_contexts:
                if str(con) in probability_map:
                    new_probs[str(con)] = probability_map[str(con)]

            top_probs = dict(sorted(new_probs.items(), key=lambda x: x[1], reverse=True)[:beam_width]) #Gets the top beam_width probabilities off the top
            string_contexts = list(top_probs.keys())
            new_contexts = []
            for con in string_contexts:
                str_values = con.strip('][').split(', ')
                new_values = []
                for val in str_values:
                    new_values.append(int(val))
                new_contexts.append(new_values)

            frontier_contexts = new_contexts
            all_contexts.append(new_contexts)
    
    return(frontier_contexts)

def efn_search(get_logits, context, max_contexts=1000, max_expansions=50, beam_width=3):

    frontier_contexts = [context]
    all_contexts = [context]
    probability_map = {}
    num_expansions = 0

    while(True):
        #Set end conditions for loop
        if (len(all_contexts) >= max_contexts):
            break
        if (num_expansions >= max_expansions):
            break

        #Get the highest probability context still available
        max_key = str(frontier_contexts[0])
        if (bool(probability_map)):
            max_key = max(probability_map.keys(), key=(lambda k: probability_map[k]))
        else: #If there are no values, then set the probability of getting the current context to 0
            probability_map[str(context)] = 0

        #Find the highest probability context
        string_context = max_key
        str_context_array = string_context.strip('][').split(', ')
        current_context = []
        for val in str_context_array:
            current_context.append(int(val))
        
        out_logits = get_logits(current_context)
        num_expansions += 1

        #Normalize the outputs (Hopefully they have already been normalized, but just in case)
        out_logits = out_logits - np.max(out_logits)
        eo_logits = np.exp(out_logits) + 1e-20
        out_logits = np.log(eo_logits / (np.sum(eo_logits)))

        #Get the top beam_width number of output contexts based on logits
        logit_indices = []
        logit_probs = []
        for _ in range(0, beam_width):
            logit_indices.append(np.argmax(out_logits))
            logit_probs.append(np.max(out_logits))
            np.put(out_logits, np.argmax(out_logits), np.min(out_logits))

        new_contexts = []
        for i in range(len(logit_indices)):
            temp_context = current_context.copy()
            temp_context.append(logit_indices[i])
            if str(current_context) in probability_map:
                probability_map[str(temp_context)] = probability_map[str(current_context)] + logit_probs[i]
            else:
                probability_map[str(temp_context)] = logit_probs[i]

            new_contexts.append(temp_context)

        if str(current_context) in probability_map.keys():
            del probability_map[str(current_context)]
        
        for new_context in new_contexts:
            frontier_contexts.append(new_context)
            all_contexts.append(new_context)

        frontier_contexts.remove(current_context)
    
    return frontier_contexts