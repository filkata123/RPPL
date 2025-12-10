def probability_model(num_neigbors):
    prob_success = 0.9
    if num_neigbors > 1:
        prob_stay = 0.05
        prob_other = (1 - prob_success - prob_stay) / (num_neigbors - 1) # we should not include the edge we want to take
    else:
        prob_stay = 0.1
        prob_other = 0.0
    return prob_success, prob_stay, prob_other