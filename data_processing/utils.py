import time
def cycle_no_memory(iterable):
    '''Copy of cycle from itertools but with no memory'''
    while True:
        for element in iterable:
            yield element

def cycle_with(leader, follower):
    #Itterator performance is good, but slow with data-loaders
#if leader is not None:
    follower_cycle = cycle_no_memory(follower)
    for element in leader:
        yield (element, next(follower_cycle))
# else:
#     for element in follower:
#         yield (None,element)


if __name__ == "__main__":
    #Test the speed of the itterator
    hi = [i for i in range(1000)]
    hello = [i for i in range(50)]
    t0 = time.time()
    for elt in cycle_with(hi, hello):
        print(elt)
    t1 = time.time()
    print(t1-t0)
    to = time.time()
    for elt in hi:
        pass
    for elt in hello:
        pass
    t1 = time.time()

    print(t1 - t0)
