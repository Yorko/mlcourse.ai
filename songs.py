animals = { 'a': ['aardvark'], 'b': ['baboon'], 'c': ['coati']}

animals['d'] = ['donkey']
animals['d'].append('dog')

def biggest(aDict):
    for k in aDict:
        if len(aDict[k]) == max(map(len,aDict.values())):
          print(k) 
biggest(animals)