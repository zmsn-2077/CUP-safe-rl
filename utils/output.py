import sys

def println(*args):
    """
    Print and flush
    """
    print(*args)
    sys.stdout.flush()

# def println(**kwargs):
#     for k,v in kwargs.items():
#         print("%s %s"%(k,v))