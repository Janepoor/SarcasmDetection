# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 19:02:03 2016

@author: Murali
"""

words = set()
with open(r'D:\IE Project\full_words.txt') as f:
    for line in f:
        words.add(line.strip())

solutions = {}
def find_words(instring):
    # First check if instring is in the dictionnary
    if instring in words:
        #print instring+" found in dict"
        return [instring]
    # No... But maybe it's a result we already computed
    if instring in solutions:
        #print instring+" found in previous solutions"
        return solutions[instring]
    # Nope. Try to split the string at all position to recursively search for results
    best_solution = None
    for i in range(1, len(instring) - 1):
        #print "instring[:i] "+instring[:i]
        #print "instring[i:] "+instring[i:]
        part1 = find_words(instring[:i])
        part2 = find_words(instring[i:])
        # Both parts MUST have a solution
        #print "part1 "+str(part1)
        #print "part2 "+str(part2)
        if part1 is None or part2 is None:
            #print "one of them is none"
            continue
        solution = part1 + part2
        #print "solution " + str(solution)
        # Is the solution found "better" than the previous one?
        if best_solution is None or len(solution) < len(best_solution):
            best_solution = solution
        #print best_solution    
    # Remember (memoize) this solution to avoid having to recompute it
    solutions[instring] = best_solution
    #print best_solution
    return best_solution
    
    
#result = find_words("thereismassesoftextinformationofpeoplescommentswhichisparsedfromhtmlbuttherearenodelimitedcharactersinthemforexamplethumbgreenappleactiveassignmentweeklymetaphorapparentlytherearethumbgreenappleetcinthestringialsohavealargedictionarytoquerywhetherthewordisreasonablesowhatsthefastestwayofextractionthxalot")
#result = find_words("tabletgod")
#assert(result is not None)
#print ' '.join(result)    