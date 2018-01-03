clean = open('abbreviations.txt').read().replace('\n', ', ').lower()
Abb_dict = dict(item.split(":") for item in clean.split(","))
print (Abb_dict)
