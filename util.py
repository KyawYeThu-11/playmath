import re

np_dict = {
    'sin':'np.sin',
    'cos':'np.cos',
    'tan':'np.tan',
    'exp':'np.exp',
    'sqrt':'np.sqrt',
}

def numpy_power(eq):
    pattern = re.compile(r"([0-9]|x|y)\*\*([0-9]|x|y)+")
    result = re.findall(pattern, eq)
    if result:
        base, exp = result[0]
        return pattern.sub(f'np.power({base}, {exp})', eq)
    else:
        return eq

def numpy_log(eq):
    pattern = re.compile(r"log\(([0-9]|x|y), (10|2|E)\)")
    result = re.findall(pattern, eq)
    if result: 
        argument, value = result[0]
        if value == '10':
            return pattern.sub(f'np.log10({argument})', eq) 
        elif value == '2':
            return pattern.sub(f'np.log2({argument})', eq) 
        else:
            return pattern.sub(f'np.log({argument})', eq)
    else:
        return eq 

def replace_functions(text):
  # Create a regular expression  from the dictionary keys
  regex = re.compile("(%s)" % "|".join(map(re.escape, np_dict.keys())))

  # For each match, look-up corresponding value in dictionary
  eq = regex.sub(lambda mo: np_dict[mo.string[mo.start():mo.end()]], text) 
  eq = numpy_power(eq)
  eq = numpy_log(eq)
  return eq