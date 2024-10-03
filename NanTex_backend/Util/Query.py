def query(self, value, type = None):
        if type is not None:
            return (type(input(value)))
        else:
            return (input(value))

def Compl_query(self, name):
    return self.query(("Please enter value for " + str(name)), type=float)