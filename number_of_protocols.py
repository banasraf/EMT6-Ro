import math

"""
Method for computing the number of all protocols.
"""
def number_of_all_protocols():
    ret = math.factorial(59) / math.factorial(19) / math.factorial(40)
    for i in range(21, 41):
        ret = ret - 2 * (math.factorial(59 - i - 1) / math.factorial(18) / math.factorial(40 - i))
        ret = ret - (59 - i - 1) * math.factorial(59 - i - 2) / math.factorial(40 - i) / math.factorial(17)
    return ret

"""
Method for computing the number of protocols for a given time period (number of 30minutes), 
number_of_30min - number of 30-minute slots that can be allocated
minimal - the minimal time difference (number of 30 minute slots) between 2 consecutive doses
max - the maximal time difference (number of 30 minute slots) between 2 consecutive doses
doses - number of doses (of an equal size) that should be distributed  
"""
def number_of_protocols_in(number_of_30min, minimal, max, doses):
    if number_of_30min < minimal:
        return 0
    else:
        ret = 0
        if doses == 1:
            return min(max, number_of_30min) - minimal + 1
        for i in range(minimal, max + 1):
            ret = ret + number_of_protocols_in(number_of_30min - i, minimal, max, doses - 1)
        return ret

"""
Number of possible protocols within 5 days (240 x 30 min), assuming that the first dose is distributed at time 0,
the minimal time difference is 4h (8 x 30min), the maximal time difference is 26h (52 x 30min),
the remaining number of doses (1.25Gy) is 7 
"""
def number_of_protocols_bmi():
    ret = number_of_protocols_in(240, 8, 52, 7)
    return ret

"""
Number of possible protocols within 5 days (240 x 30 min), assuming that the first dose is distributed at time 0,
the minimal time difference is 10h (20 x 30min), the maximal time difference is 32h (64 x 30min),
the remaining number of doses is 4 
"""
def number_of_protocols_bmii():
    ret = number_of_protocols_in(240, 20, 64, 4)
    return ret


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print(number_of_all_protocols())
    print(number_of_protocols_bmi())
    print(number_of_protocols_bmii())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
