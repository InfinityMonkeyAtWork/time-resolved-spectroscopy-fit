#
# miscalleneous functions
#

#
def dict_info(dictionary, dict_element=True, list_element=True):
    """
    Print all items and their type of the input <dictionary>
    If values are dictionaries or lists they will be dissected as well
    """
    for key, value in dictionary.items():
        print()
        print(key)
        print(type(key))
        print()
        print(value)
        print(type(value))
        if isinstance(value, dict):
            if dict_element==True:
                dict_info(value)
        elif isinstance(value, list):
            if list_element==True:
                list_info(value)
    #
    return None

#
def list_info(listIN, dict_element=True, list_element=True):
    """
    Print each element of a list as well as the type of the element
    If element is a list or dict it will be dissected as well
    """
    for element in listIN:
        if isinstance(element, dict):
            if dict_element==True:
                dict_info(element)
        elif isinstance(element, list):
            if list_element==True:
                list_info(element)
        else:
            print(element)
            print(type(element))
    print()
    #
    return None

#
#
#