import itertools

def chunked(it, size):
    it = iter(it)
    while True:
        p = tuple(itertools.islice(it, size))
        if not p:
            break
        yield p

batch_size = 2
thisdict = {
  "brand": "Ford",
  "electric": False,
  "year": 1964,
  "colors": ["red", "white", "blue"]
}
for chunk in chunked(thisdict.items(), batch_size):
    for key, val in chunk:
        print(key,val)