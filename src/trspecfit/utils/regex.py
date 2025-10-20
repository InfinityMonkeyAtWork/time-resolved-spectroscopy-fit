import re

#
def get_index_of_matches(string, pattern):
    """
    Find <pattern> in <string> and return a list containing elements
    [start_index, stop_index]
    """
    #
    return [m.span() for m in re.finditer(pattern, string)]

#
def get_match_number_N(N, inds, string, include_match_itself=True):
    """
    Get <N>th match within <string> where <inds> is returned by the 
    <get_index_of_matches(string, pattern)> method
    <include_match_itself> at the beginning of match? True/False
    """
    if include_match_itself == True: s=0
    elif include_match_itself == False: s=1
    
    if N > len(inds):
        print(f'Selected match {N} exceeds number of matches!')
        print(f'(length of <inds> {len(inds)})')
        return ''
    elif N < len(inds): # standard case
        return string[inds[N][s] : inds[N+1][0]]
    elif N == len(inds): # get last match
        return string[inds[N][s] : ]

#
def search_line_by_line(lines, str_search, location,
                        include_str_search=False, print_info=0):
    """
    Returns the line(s) that contain search string.

    <lines> is the output from a str.splitlines() call
    <str_search> is the string to be searched
    <location> can be one of "start", "end", "anywhere"
    <include_str_search> = True/False
    (works for location 'start' and 'end')
    """
    matches = [] # place holder for all lines that match

    for line in lines:

        if print_info >= 1: print(line)

        if location == 'start':
            if str_search == line[:len(str_search)]:
                if include_str_search == True:
                    matches.append(line)
                elif include_str_search == False:
                    matches.append(line[len(str_search):])

        elif location == 'end':
            if str_search == line[-len(str_search):]:
                if include_str_search == True:
                    matches.append(line)
                elif include_str_search == False:
                    matches.append(line[:len(str_search)])

        elif location == 'anywhere':
            if str_search in line:
                matches.append(line) # always include the match itself
    #
    return matches

#
# LEGACY [probably this is all re-inventing the wheel]
#
# string, text, regular expressions: search and print functions
#
def find_all_subs_in_str(string, search_str, overlap, case):
    """
    this function finds all instances of <search_str> in <string> and 
    returns a list of indices of where the matches start in <string>. 
    to find overlapping matches set overlap to 1, otherwise 0.
    case: 0 -> ignore case, 1 -> consider case
    
    use like so: (WATCH OUT FOR "list")
    list_ind = list(find_all_subs_in_str(text_str,
                                         search_str,
                                         overlap,
                                         case)
                    )
    """
    ind = 0
    # convert string and search string to lower case 
    if case == 0:
        string = string.lower()
        search_str = search_str.lower()
    while True:
        temp = re.search(search_str, string[ind:])
        if temp == None: return
        yield [temp.span()[0] +ind, temp.span()[1] +ind]
        if overlap == 0: # non-overlapping matches, faster:
            ind += temp.span()[1]
        elif overlap == 1: # find overlapping matches, slower:
            ind += temp.span()[0] +1

#
def print_all_delta(string, list_ind, delta):
    """
    print text from string around instances of <list_ind> in 
    <string> with the surrounding <delta> number of characters 
    each (plus/minus).
    
    use like so:
    print_all_delta(string_IN, list_ind, delta)
    """
    print("Number of instances found is: " +str(len(list_ind)))
    for instance in range(len(list_ind)):
        print()
        print("Index of instance: " +str(list_ind[instance]))
        print(string[list_ind[instance]-delta : list_ind[instance]+delta])

#
def find_print_str(string, search_str, overlap, delta):
    """
    this function finds all instances of <search_str> in <string>
    and returns a list of indices of where the matches start in 
    <string>. to find overlapping matches, which is slower, set 
    overlap to 1, otherwise 0. print all instances of <search_str>
    in <string> with the surrounding <delta> characters each.
    
    use like so:
    find_print_str(string, search_str, overlap, delta)
    """
    list_ind = list(find_all_subs_in_str(string, search_str, overlap))
    print("Number of instances found is: " +str(len(list_ind)))
    for instance in range(len(list_ind)):
        print()
        print("Index of instance: " +str(list_ind[instance]))
        print(string[list_ind[instance]-delta : list_ind[instance]+delta])
#
#
#