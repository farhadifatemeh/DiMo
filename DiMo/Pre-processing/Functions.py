#########################################################FUNCTIONS####################
def listToString(s):
    # initialize an empty string
    str1 = ""

    # traverse in the string
    for ele in s:
        str1 += ele

        # return string
    return str1
    ###########################################


def occurrences(string, sub):
    count = start = 0
    while True:
        start = string.find(sub, start) + 1
        if start > 0:
            count += 1
        else:
            return count


############
def absSecond(val):
    return abs(int(val[1]) - int(val[2]))


#############
def fiveSecond(val):
    return float(val[5])


## /// Function of Weights
def binomial(val):
    x = abs(int(val))
    b = 0.03125 * (x * x) + (1.25 * x)
    return b
